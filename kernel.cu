#ifndef _KERNEL_CU_
#define _KERNEL_CU_

#include <cutil_inline.h>
#include <cutil_math.h>

typedef unsigned int uint;
typedef unsigned short ushort;
typedef struct 
{
    float4 m[3];
} float3x4;

__constant__ float3x4 invViewMatrix;
__constant__ cudaExtent dataDim;
__constant__ uint nTriangles;
__constant__ float3 geomSize;
__constant__ char geomType;
__constant__ bool geomIsTriangles;

cudaArray *dataArray = 0;
texture<ushort,  3, cudaReadModeNormalizedFloat> dataTex;

cudaArray *trianglesArray = 0;
texture<float4, 1, cudaReadModeElementType> trianglesTex;

cudaArray *transferFuncArray;
texture<float4, 1, cudaReadModeElementType> transferTex;


__device__
float pi()
{
	return 4.0f * atanf(1.0f);
}

__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

__device__
float det(float2 u, float2 v)
{
	return (u.x * v.y) - (u.y * v.x);
}

__device__
float3 cylToCart(float3 point)
{
	return make_float3(point.x * cosf(point.y), point.z, -point.x * sinf(point.y));
}

__device__
float3 cartToCyl(const float3 &point)
{
	float angle = atan2f(-point.z, point.x);
	if (angle < 0)
	{
		angle += 8.0f * atanf(1.0f);
	}
	return make_float3(sqrtf((point.z * point.z) + (point.x * point.x)), angle, point.y);
}

__device__
float3 spherToCart(float3 point)
{
	return make_float3(point.x * cosf(point.z) * cosf(point.y), point.x * sinf(point.z), -point.x * cosf(point.z) * sinf(point.y));
}

__device__
float3 cartToSpher(const float3 &point)
{
	float r = sqrtf((point.z * point.z) + (point.x * point.x));
	float R = sqrtf((point.x * point.x) + (point.y * point.y) + (point.z * point.z));
	float longit = atan2f(-point.z, point.x);
	if (longit < 0)
	{
		longit += 2.0f * pi();
	}
	float lat = atan2f(point.y, r);
	return make_float3(R, longit, lat);
}

__device__
float3 baryTriangleCoefs(const float3 &P0, const float3 &P1, const float3 &P2, const float3 &point)
{
	// (u, v) orthonormal basis of the triangle plane
	float3 u = normalize(P1 - P0);
	float3 v = normalize(cross(cross(u, P2 - P0), u));

	float2 P0P = {dot(point - P0, u), dot(point - P0, v)};
	float2 P1P = {dot(point - P1, u), dot(point - P1, v)};
	float2 P2P = {dot(point - P2, u), dot(point - P2, v)};

	float A = det(P1P, P2P);
	float B = det(P2P, P0P);
	float C = det(P0P, P1P);

	float delta = A + B + C;
	return make_float3(A / delta, B / delta, C / delta);
}

__device__
float3 baryTriangleCoefs(uint firstInd, float3 point)
{
	return baryTriangleCoefs(make_float3(tex1D(trianglesTex, firstInd)), make_float3(tex1D(trianglesTex, firstInd + 1)), make_float3(tex1D(trianglesTex, firstInd + 2)), point);
}

__device__
bool inTriangle(float3 baryCoefs)
{
	return baryCoefs.x >= 0 && baryCoefs.y >= 0 && baryCoefs.z >= 0;
}

__device__
bool rayIntersectTriangles(const float3 &rayOrig, const float3 &rayVect, float* inAbs, float* outAbs)
{
	bool inAbsFound = false;
	bool outAbsFound = false;
	float inAbsVal, outAbsVal;
	for (uint i = 0; i < nTriangles; i++) {
		float3 P0 = make_float3(tex1D(trianglesTex, 3 * i));
		float3 P1 = make_float3(tex1D(trianglesTex, (3 * i) + 1));
		float3 P2 = make_float3(tex1D(trianglesTex, (3 * i) + 2));
		float3 n = cross(P1 - P0, P2 - P0);
		float d = dot(n, rayVect);
		if (d > 0)
		{
			float s = dot(n, P0 - rayOrig) / d;
			float3 baryCoefs = baryTriangleCoefs(3 * i, rayOrig + (s * rayVect));
			if (inTriangle(baryCoefs)) 
			{
				outAbsVal = outAbsFound ? fmaxf(outAbsVal, s) : s;
				outAbsFound = true;
			}
		}
		if (d < 0)
		{
			float s = dot(n, P0 - rayOrig) / d;
			float3 baryCoefs = baryTriangleCoefs(3 * i, rayOrig + (s * rayVect));
			if (inTriangle(baryCoefs)) 
			{
				inAbsVal = inAbsFound ? fminf(inAbsVal, s) : s;
				inAbsFound = true;
			}
		}
	}

	if (inAbsFound && !outAbsFound)
	{
		outAbsVal = inAbsVal + 2 * geomSize.x / length(rayVect);
	}
	if (!inAbsFound && outAbsFound)
	{
		inAbsVal = outAbsVal - 2 * geomSize.x / length(rayVect);
	}

	*inAbs = inAbsVal;
	*outAbs = outAbsVal;

	return outAbsVal >= inAbsVal && outAbsFound && inAbsFound;
}

__device__
bool rayIntersectCyl(const float3 &rayOrig, const float3 &rayVect, float* inAbs, float* outAbs)
{
	float inAbsVal = 0;
	float outAbsVal = 0;
	bool inAbsFound = false;
	bool outAbsFound = false;

	float3 lastPoint = cylToCart(geomSize);

	// Front rectangle
	float3 P0 = {0, 0, 0};
	float3 P1 = {geomSize.x, 0, 0};
	float3 P2 = {0, geomSize.z, 0};
	float3 n = cross(P1 - P0, P2 - P0);
	float d = dot(n, rayVect);
	if (d != 0)
	{
		float s = dot(n, P0 - rayOrig) / d;
		float3 P = rayOrig + s * rayVect;
		if (P.x >= 0 && P.x <= geomSize.x && P.y >= 0 && P.y <= geomSize.z)
		{
			if (d > 0)
			{
				outAbsVal = outAbsFound ? fmaxf(outAbsVal, s) : s;
				outAbsFound = true;
			}
			else
			{
				inAbsVal = inAbsFound ? fminf(inAbsVal, s) : s;
				inAbsFound = true;
			}
		}
	}

	// Back rectangle
	P0 = make_float3(lastPoint.x, 0, lastPoint.z);
	P1 = make_float3(0, 0, 0);
	P2 = make_float3(0, geomSize.z, 0);
	n = cross(P1 - P0, P2 - P0);
	d = dot(n, rayVect);
	if (d != 0)
	{
		float s = dot(n, P0 - rayOrig) / d;
		float3 P = rayOrig + s * rayVect;
		float x = dot(P, P0 - P1) / length(P0 - P1);
		if (x >= 0 && x <= geomSize.x && P.y >= 0 && P.y <= geomSize.z)
		{
			if (d > 0)
			{
				outAbsVal = outAbsFound ? fmaxf(outAbsVal, s) : s;
				outAbsFound = true;
			}
			else
			{
				inAbsVal = inAbsFound ? fminf(inAbsVal, s) : s;
				inAbsFound = true;
			}
		}
	}

	// Top face
	P0 = make_float3(0, geomSize.z, 0);
	n = make_float3(0, 1, 0);
	d = dot(n, rayVect);	
	if (d != 0)
	{
		float s = dot(n, P0 - rayOrig) / d;
		float3 P = cartToCyl(rayOrig + s * rayVect);
		if (P.x <= geomSize.x && P.y <= geomSize.y)
		{
			if (d > 0)
			{
				outAbsVal = outAbsFound ? fmaxf(outAbsVal, s) : s;
				outAbsFound = true;
			}
			else
			{
				inAbsVal = inAbsFound ? fminf(inAbsVal, s) : s;
				inAbsFound = true;
			}
		}
	}

	// Bottom face
	P0 = make_float3(0, 0, 0);
	n = make_float3(0, -1, 0);
	d = dot(n, rayVect);		
	if (d != 0)
	{
		float s = dot(n, P0 - rayOrig) / d;
		float3 P = cartToCyl(rayOrig + s * rayVect);
		if (P.x <= geomSize.x && P.y <= geomSize.y)
		{
			if (d > 0)
			{
				outAbsVal = outAbsFound ? fmaxf(outAbsVal, s) : s;
				outAbsFound = true;
			}
			else
			{
				inAbsVal = inAbsFound ? fminf(inAbsVal, s) : s;
				inAbsFound = true;
			}
		}
	}

	// Lateral face
	float2 u = {rayVect.x, rayVect.z};
	float2 v = {rayOrig.x, rayOrig.z};
	float a = dot(u, u);
	float b = 2 * dot(u, v);
	float c = dot(v, v) - (geomSize.x * geomSize.x);
	float delta = b * b - 4 * a * c;
	if (delta >= 0)
	{
		float s[2] = {(-b - sqrtf(delta)) / (2 * a), (-b + sqrtf(delta)) / (2 * a)};
		float3 P[2] = {rayOrig + s[0] * rayVect, rayOrig + s[1] * rayVect};

		float3 PCyl[2] = {cartToCyl(P[0]), cartToCyl(P[1])};
		float d[2] = {P[0].x * rayVect.x + P[0].z * rayVect.z, P[1].x * rayVect.x + P[1].z * rayVect.z};
		bool PInFace[2] = {PCyl[0].y <= geomSize.y && PCyl[0].z >= 0 && PCyl[0].z <= geomSize.z, PCyl[1].y <= geomSize.y && PCyl[1].z >= 0 && PCyl[1].z <= geomSize.z};

		for (uint i = 0; i <= 1; i++)
		{
			if (PInFace[i])
			{
				if(d[i] > 0)
				{
					outAbsVal = outAbsFound ? fmaxf(outAbsVal, s[i]) : s[i];
					outAbsFound = true;
				}
				if(d[i] < 0)
				{
					inAbsVal = inAbsFound ? fminf(inAbsVal, s[i]) : s[i];
					inAbsFound = true;
				}
				if(d[i] == 0)
				{
					outAbsVal = outAbsFound ? fmaxf(outAbsVal, s[i]) : s[i];
					outAbsFound = true;
					inAbsVal = inAbsFound ? fminf(inAbsVal, s[i]) : s[i];
					inAbsFound = true;
				}
			}
		}
	}

	if (inAbsFound && !outAbsFound)
	{
		outAbsVal = inAbsVal + 2 * geomSize.x / length(rayVect);
	}
	if (!inAbsFound && outAbsFound)
	{
		inAbsVal = outAbsVal - 2 * geomSize.x / length(rayVect);
	}

	*inAbs = inAbsVal;
	*outAbs = outAbsVal;

	return outAbsVal >= inAbsVal && outAbsFound && inAbsFound;
}



__device__
bool rayIntersectSpher(const float3 &rayOrig, const float3 &rayVect, float* inAbs, float* outAbs)
{
	float inAbsVal = 0;
	float outAbsVal = 0;
	bool inAbsFound = false;
	bool outAbsFound = false;

	float3 lastPoint = spherToCart(geomSize);

	// Top and bottom faces
	float2 u = {rayVect.x, rayVect.z};
	float2 v = {rayOrig.x, rayOrig.z};
	float t2 = tanf((pi() / 2.0f) - geomSize.z);
	t2 = t2 * t2;
	float a = dot(u, u) - (rayVect.y * rayVect.y * t2);
	float b = 2 * (dot(u, v) - (rayVect.y * rayOrig.y * t2));
	float c = dot(v, v) - (rayOrig.y * rayOrig.y * t2);
	float delta = b * b - 4 * a * c;
	if (delta >= 0)
	{
		float s[2] = {(-b - sqrtf(delta)) / (2 * a), (-b + sqrtf(delta)) / (2 * a)};
		float3 P[2] = {rayOrig + s[0] * rayVect, rayOrig + s[1] * rayVect};

		float3 PSpher[2] = {cartToSpher(P[0]), cartToSpher(P[1])};
		float d[2] = {(PSpher[0].z / fabs(PSpher[0].z)) * dot(make_float3(-sinf(PSpher[0].z) * cosf(PSpher[0].y), cosf(PSpher[0].z), sinf(PSpher[0].z) * sinf(PSpher[0].y)), rayVect),
					  (PSpher[1].z / fabs(PSpher[1].z)) * dot(make_float3(-sinf(PSpher[1].z) * cosf(PSpher[1].y), cosf(PSpher[1].z), sinf(PSpher[1].z) * sinf(PSpher[1].y)), rayVect)};
		bool PInFace[2] = {PSpher[0].y <= geomSize.y && PSpher[0].x <= geomSize.x, PSpher[1].y <= geomSize.y && PSpher[1].x <= geomSize.x};

		for (uint i = 0; i <= 1; i++)
		{
			if (PInFace[i])
			{
				if(d[i] > 0)
				{
					outAbsVal = outAbsFound ? fmaxf(outAbsVal, s[i]) : s[i];
					outAbsFound = true;
				}
				if(d[i] < 0)
				{
					inAbsVal = inAbsFound ? fminf(inAbsVal, s[i]) : s[i];
					inAbsFound = true;
				}
				if(d[i] == 0)
				{
					outAbsVal = outAbsFound ? fmaxf(outAbsVal, s[i]) : s[i];
					outAbsFound = true;
					inAbsVal = inAbsFound ? fminf(inAbsVal, s[i]) : s[i];
					inAbsFound = true;
				}
			}
		}
	}

	// Front face
	float3 P0 = make_float3(0, 0, 0);
	float3 n = make_float3(0, 0, 1);
	float d = dot(n, rayVect);	
	if (d != 0)
	{
		float s = dot(n, P0 - rayOrig) / d;
		float3 P = rayOrig + s * rayVect;
		float3 PSpher = cartToSpher(P);
		if (PSpher.x <= geomSize.x && fabs(PSpher.z) <= geomSize.z && P.x >= 0)
		{
			if (d > 0)
			{
				outAbsVal = outAbsFound ? fmaxf(outAbsVal, s) : s;
				outAbsFound = true;
			}
			else
			{
				inAbsVal = inAbsFound ? fminf(inAbsVal, s) : s;
				inAbsFound = true;
			}
		}
	}

	// Back face
	float3 P1 = lastPoint;
	float3 P2 = make_float3(lastPoint.x, -lastPoint.y, lastPoint.z);
	n = cross(P1 - P0, P2 - P0);
	d = dot(n, rayVect);	
	if (d != 0)
	{
		float s = dot(n, P0 - rayOrig) / d;
		float3 P = rayOrig + s * rayVect;
		float3 PSpher = cartToSpher(P);
		if (PSpher.x <= geomSize.x && fabs(PSpher.z) <= geomSize.z && dot(P, P2 - P0) >= 0)
		{
			if (d > 0)
			{
				outAbsVal = outAbsFound ? fmaxf(outAbsVal, s) : s;
				outAbsFound = true;
			}
			else
			{
				inAbsVal = inAbsFound ? fminf(inAbsVal, s) : s;
				inAbsFound = true;
			}
		}
	}

	// Lateral face
	a = dot(rayVect, rayVect);
	b = 2 * dot(rayVect, rayOrig);
	c = dot(rayOrig, rayOrig) - (geomSize.x * geomSize.x);
	delta = b * b - 4 * a * c;
	if (delta >= 0)
	{
		float s[2] = {(-b - sqrtf(delta)) / (2 * a), (-b + sqrtf(delta)) / (2 * a)};
		float3 P[2] = {rayOrig + s[0] * rayVect, rayOrig + s[1] * rayVect};

		float3 PSpher[2] = {cartToSpher(P[0]), cartToSpher(P[1])};
		float d[2] = {dot(P[0], rayVect), dot(P[1], rayVect)};
		bool PInFace[2] = {PSpher[0].y <= geomSize.y && fabs(PSpher[0].z) <= geomSize.z, PSpher[1].y <= geomSize.y && fabs(PSpher[1].z) <= geomSize.z};

		for (uint i = 0; i <= 1; i++)
		{
			if (PInFace[i])
			{
				if(d[i] > 0)
				{
					outAbsVal = outAbsFound ? fmaxf(outAbsVal, s[i]) : s[i];
					outAbsFound = true;
				}
				if(d[i] < 0)
				{
					inAbsVal = inAbsFound ? fminf(inAbsVal, s[i]) : s[i];
					inAbsFound = true;
				}
				if(d[i] == 0)
				{
					outAbsVal = outAbsFound ? fmaxf(outAbsVal, s[i]) : s[i];
					outAbsFound = true;
					inAbsVal = inAbsFound ? fminf(inAbsVal, s[i]) : s[i];
					inAbsFound = true;
				}
			}
		}
	}
	if (inAbsFound && !outAbsFound)
	{
		outAbsVal = inAbsVal + 2 * geomSize.x / length(rayVect);
	}
	if (!inAbsFound && outAbsFound)
	{
		inAbsVal = outAbsVal - 2 * geomSize.x / length(rayVect);
	}

	*inAbs = inAbsVal;
	*outAbs = outAbsVal;

	return outAbsVal >= inAbsVal && (outAbsFound || inAbsFound);
}

__device__
float3 cubicInterpMixture(float x)
{
	float i = floor(x);
	float a = x - i;
	float a2 = a * a;
	float a3 = a * a2;
	float w0 = -a3 + (2 * a2) - a;
	float w1 = a3 - (2 * a2) + 1;
	float w2 = -a3 + a2 + a;
	float w3 = a3 - a2;
	return make_float3(i - (w0 / (w0 + w1)), i + 1 + (w3 / (w2 + w3)), w2 + w3);
}

__device__
float cubicInterpolate(const float3 &texCoord)
{
	float3 mixX = cubicInterpMixture(texCoord.x);
	float3 mixY = cubicInterpMixture(texCoord.y);
	float3 mixZ = cubicInterpMixture(texCoord.z);

	float vX0Y0Z0 = (float) tex3D(dataTex, mixX.x, mixY.x, mixZ.x);
	float vX0Y1Z0 = (float) tex3D(dataTex, mixX.x, mixY.y, mixZ.x);
	float vX0Y0Z1 = (float) tex3D(dataTex, mixX.x, mixY.x, mixZ.y);
	float vX0Y1Z1 = (float) tex3D(dataTex, mixX.x, mixY.y, mixZ.y);
	float vX1Y0Z0 = (float) tex3D(dataTex, mixX.y, mixY.x, mixZ.x);
	float vX1Y1Z0 = (float) tex3D(dataTex, mixX.y, mixY.y, mixZ.x);
	float vX1Y0Z1 = (float) tex3D(dataTex, mixX.y, mixY.x, mixZ.y);
	float vX1Y1Z1 = (float) tex3D(dataTex, mixX.y, mixY.y, mixZ.y);

	float vY0Z0 = lerp(vX0Y0Z0, vX1Y0Z0, mixX.z);
	float vY1Z0 = lerp(vX0Y1Z0, vX1Y1Z0, mixX.z);
	float vZ0 = lerp(vY0Z0, vY1Z0, mixY.z);

	float vY0Z1 = lerp(vX0Y0Z1, vX1Y0Z1, mixX.z);
	float vY1Z1 = lerp(vX0Y1Z1, vX1Y1Z1, mixX.z);
	float vZ1 = lerp(vY0Z1, vY1Z1, mixY.z);

	return lerp(vZ0, vZ1, mixZ.z);
}


__device__
float interpolateCyl(const float3 &point, float outVal, int triCubic)
{
	float lengthStep = geomSize.x / dataDim.width;
	float angleStep = geomSize.y / dataDim.height;
	float heightStep = geomSize.z / dataDim.depth;

	float3 cylCoord = cartToCyl(point);
	float3 texCoord = cylCoord / make_float3(lengthStep, angleStep, heightStep);

	if (texCoord.x < 0 || texCoord.x > dataDim.width - 1 || texCoord.y < 0 ||  texCoord.y > dataDim.height - 1 || texCoord.z < 0 ||  texCoord.z > dataDim.depth - 1)
	{
		return outVal;
	}
	if (triCubic)
	{
		return cubicInterpolate(texCoord);
	}
	return (float) tex3D(dataTex, texCoord.x, texCoord.y, texCoord.z);
}

__device__
float interpolateSpher(const float3 &point, float outVal, int triCubic)
{
	float lengthStep = geomSize.x / dataDim.depth;
	float longStep = geomSize.y / dataDim.width;
	float latStep = 2 * geomSize.z / dataDim.height;

	float3 spherCoord = cartToSpher(point) + make_float3(0, 0, geomSize.z);
	float3 texCoord = make_float3(spherCoord.y / longStep, spherCoord.z / latStep, spherCoord.x / lengthStep);

	if (texCoord.x < 0 || texCoord.x > dataDim.width - 1 || texCoord.y < 0 ||  texCoord.y > dataDim.height - 1 || texCoord.z < 0 ||  texCoord.z > dataDim.depth - 1)
	{
		return outVal;
	}
	if (triCubic)
	{
		return cubicInterpolate(texCoord);
	}
	return (float) tex3D(dataTex, texCoord.x, texCoord.y, texCoord.z);
}

__device__
float interpolate(float3 point, float outVal, int triCubic)
{
	switch(geomType)
	{
	case 0:
		return interpolateCyl(point, outVal, triCubic);
	case 1:
		return interpolateSpher(point, outVal, triCubic);
	}
}

__device__
bool rayIntersect(float3 rayOrig, float3 rayVect, float* inAbs, float* outAbs)
{
	if (geomIsTriangles)
	{
		return rayIntersectTriangles(rayOrig, rayVect, inAbs, outAbs);
	}
	switch(geomType)
	{
	case 0:
		return rayIntersectCyl(rayOrig, rayVect, inAbs, outAbs);
	case 1:
		return rayIntersectSpher(rayOrig, rayVect, inAbs, outAbs);
	}
}

__device__ 
uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__global__ 
void kernel(uint* output, uint dispWidth, uint dispHeight, float step, int triCubic, float density, float regBrightness, float xrayBrightness, float isoValue, float transferScale, int compoType)
{

	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    float u = (2.0f * x / (float) dispWidth) - 1.0f;
    float v = (2.0f * y / (float) dispHeight) - 1.0f;

	float3 O = make_float3(mul(invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    float3 vect = normalize(make_float3(u, v, -2.0f));
    vect = mul(invViewMatrix, vect);

	float s = sqrtf(u * u + v * v + 4.0f) / 2.0f;

	float inAbs, outAbs;
	bool ok = rayIntersect(O, vect, &inAbs, &outAbs);


	if (ok && outAbs >= s)
	{
		inAbs = fmaxf(s, inAbs);

		switch(compoType)
		{
		case 0:
			float4 sum = make_float4(0.0f);
			for (float s = outAbs; s >= inAbs; s -= step)	
			{
				float sample = interpolate(O + s * vect, 0, triCubic);

				if (sample >= isoValue)
				{
					// Lookup in transfer function texture
					float4 col = tex1D(transferTex, sample * transferScale);

					// Accumulate result
					sum = lerp(sum, col, col.w * density);
				}
			}

			if ((x < dispWidth) && (y < dispHeight)) {
				// Write output color
				uint i = __umul24(y, dispWidth) + x;
				output[i] = rgbaFloatToInt(sum * regBrightness);
			}
			break;

		case 1:
			float maxVal = 0.0f;
			for (float s = outAbs; s >= inAbs; s -= step)	
			{
				float sample = interpolate(O + s * vect, 0, triCubic);
				if (sample >= isoValue)
				{
					maxVal = fmaxf(maxVal, sample);
				}
			}
			maxVal *= xrayBrightness;

			if ((x < dispWidth) && (y < dispHeight)) {
				// Write output color
				uint i = __umul24(y, dispWidth) + x;
				output[i] = rgbaFloatToInt(make_float4(maxVal, maxVal, maxVal, 1.0f));
			}
		}
	}
}

extern "C"
void loadData(const ushort* data, const cudaExtent* dim)
{
	cutilSafeCall( cudaMemcpyToSymbol(dataDim, dim, sizeof(cudaExtent)) );

	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<ushort>();
    cutilSafeCall( cudaMalloc3DArray(&dataArray, &channelDesc, (*dim)) );

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*) data, (*dim).width * sizeof(ushort), (*dim).width, (*dim).height);
    copyParams.dstArray = dataArray;
    copyParams.extent = (*dim);
    copyParams.kind = cudaMemcpyHostToDevice;
    cutilSafeCall( cudaMemcpy3D(&copyParams) );

    // set texture parameters
    dataTex.normalized = false;
    dataTex.filterMode = cudaFilterModeLinear;
    dataTex.addressMode[0] = cudaAddressModeClamp;
    dataTex.addressMode[1] = cudaAddressModeClamp;
    dataTex.addressMode[2] = cudaAddressModeClamp;

    // bind array to 3D texture
    cutilSafeCall( cudaBindTextureToArray(dataTex, dataArray, channelDesc) );
}

extern "C" 
void loadTransferFunc(const float4 transferFunc[])
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cutilSafeCall( cudaMallocArray( &transferFuncArray, &channelDesc, 9, 1) ); 
    cutilSafeCall( cudaMemcpyToArray( transferFuncArray, 0, 0, transferFunc, 9 * sizeof(float4), cudaMemcpyHostToDevice) );

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;
    transferTex.addressMode[0] = cudaAddressModeClamp;

    cutilSafeCall( cudaBindTextureToArray( transferTex, transferFuncArray, channelDesc));
}

extern "C"
void loadTrianglesGeometry(const float4* triangles, const uint* n, const float3* size, const char* type)
{
	cutilSafeCall( cudaMemcpyToSymbol(nTriangles, n, sizeof(uint)) );
	cutilSafeCall( cudaMemcpyToSymbol(geomSize, size, sizeof(float3)) );
	cutilSafeCall( cudaMemcpyToSymbol(geomType, type, sizeof(char)) );
	bool isTriangles = true;
	cutilSafeCall( cudaMemcpyToSymbol(geomIsTriangles, &isTriangles, sizeof(bool)) );

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cutilSafeCall( cudaMallocArray( &trianglesArray, &channelDesc, 3 * (*n), 1) ); 
    cutilSafeCall( cudaMemcpyToArray( trianglesArray, 0, 0, triangles, 3 * (*n) * sizeof(float4), cudaMemcpyHostToDevice) );

    trianglesTex.filterMode = cudaFilterModePoint;
    trianglesTex.normalized = false;
    trianglesTex.addressMode[0] = cudaAddressModeClamp;

    cutilSafeCall( cudaBindTextureToArray( trianglesTex, trianglesArray, channelDesc) );
}

extern "C" void loadGeometry(const float3* size, const char* type)
{
	cutilSafeCall( cudaMemcpyToSymbol(geomSize, size, sizeof(float3)) );
	cutilSafeCall( cudaMemcpyToSymbol(geomType, type, sizeof(char)) );
	bool isTriangles = false;
	cutilSafeCall( cudaMemcpyToSymbol(geomIsTriangles, &isTriangles, sizeof(bool)) );
}

extern "C"
void copyInvViewMatrix(const float* matrix)
{
    cutilSafeCall( cudaMemcpyToSymbol(invViewMatrix, matrix,  sizeof(float4) * 3) );
}

extern "C"
void renderKernel(const dim3 &gridSize, const dim3 &blockSize, uint* output, uint dispWidth, uint dispHeight, float rayStep, int triCubic, float density, float regBrightness, float xrayBrightness, float isoValue, float transferScale, int compoType)
{
	kernel<<<gridSize, blockSize>>>(output, dispWidth, dispHeight, rayStep, triCubic, density, regBrightness, xrayBrightness, isoValue, transferScale, compoType);
}

extern "C" 
void freeCudaBuffers()
{
    cutilSafeCall( cudaFreeArray(dataArray) );
    cutilSafeCall( cudaFreeArray(trianglesArray) );
    cutilSafeCall( cudaFreeArray(transferFuncArray) );
}

#endif