#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <cutil_inline.h>
#include <cutil_math.h>
#include <driver_functions.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glui.h>
#include <cuda_gl_interop.h>

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

extern "C" void loadData(const ushort* data, const cudaExtent* dim);
extern "C" void loadTrianglesGeometry(const float4* triangles, const uint* n, const float3* size, const char* type);
extern "C" void loadGeometry(const float3* size, const char* type);
extern "C" void loadTransferFunc(const float4 transferFunc[]);
extern "C" void freeCudaBuffers();
extern "C" void copyInvViewMatrix(const float* matrix);
extern "C" void renderKernel(const dim3 &gridSize, const dim3 &blockSize, uint* output, uint dispWidth, uint dispHeight, float rayStep, int triCubic, float density, float regBrightness, float xrayBrightness, float isoValue, float transferScale, int compoType);


uint dispWidth, dispHeight;
int triCubic;
float rayStep, density, regBrightness, xrayBrightness, isoValue, transferScale;
int compoType;
dim3 gridSize, blockSize;
float3 viewRotation, viewTranslation;
float invViewMatrix[12];
int ox, oy;
int buttonState = 0;
GLuint pbo = 0;
uint timer = 0;
int fpsCount = 0;
uint frameCount = 0;
int mainWindow;

// Debug function
void printVect(const float3 &vect)
{
	printf("\nx: %f, y: %f, z: %f", vect.x, vect.y, vect.z);
}

// Euclidian division + 1 if not divisible
inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

inline double degToRad(double angle)
{
	return 4.0 * atan(1.0) * angle / 180.0;
}

// Render image using CUDA
void render()
{
	// Copy inverse view matrix to device
	copyInvViewMatrix(invViewMatrix);

    // Map PBO to get CUDA device pointer
    uint* output;
    cutilSafeCall( cudaGLMapBufferObject((void**) &output, pbo) );
    cutilSafeCall( cudaMemset(output, 0, dispWidth * dispHeight * 4) );

    // Call CUDA kernel, writing results to PBO

    renderKernel(gridSize, blockSize, output, dispWidth, dispHeight, rayStep, triCubic, density, regBrightness, xrayBrightness, isoValue, transferScale, compoType);

    cutilCheckMsg( "Kernel failed" );

    cutilSafeCall( cudaGLUnmapBufferObject(pbo) );
}

// Compute and display framerate
void computeFPS()
{
    frameCount++;
    fpsCount++;
    if (fpsCount == 1) {
        char fps[256];
        float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
        sprintf(fps, "Direct Curve Volume Render: %3.1f fps", ifps);
        glutSetWindowTitle(fps);
        fpsCount = 0; 
        cutilCheckError( cutResetTimer(timer) );  
    }
}

// GLUT display handler
void display()
{
    cutilCheckError( cutStartTimer(timer) ); 

    // Use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
    glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
    glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

	// Compute inverse view matrix
    invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];

    render();

    // Draw image from PBO
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(dispWidth, dispHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glutSwapBuffers();
    glutReportErrors();

	cutilCheckError( cutStopTimer(timer) ); 

    computeFPS();
}

// Program atrexit handler
void cleanup()
{
	freeCudaBuffers();
	cutilSafeCall( cudaGLUnregisterBufferObject(pbo) );    
	glDeleteBuffersARB(1, &pbo);
}

// Initiate PBO
void initPixelBuffer()
{
    if (pbo) {
        // Delete old buffer
        cutilSafeCall( cudaGLUnregisterBufferObject(pbo) );
        glDeleteBuffersARB(1, &pbo);
    }

    // Create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, dispWidth * dispHeight * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	cutilSafeCall( cudaGLRegisterBufferObject(pbo) );
}

// GLUT reshape handler
void reshape(int x, int y)
{
    dispWidth = x; 
	dispHeight = y;

	gridSize = dim3(iDivUp(dispWidth, blockSize.x), iDivUp(dispHeight, blockSize.y));

    initPixelBuffer();

    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

// GLUT/GLUI idle handler
void idle()
{
	if (glutGetWindow() != mainWindow)
	{
		glutSetWindow(mainWindow);
	}
	glutPostRedisplay();
}

// GLUT mouse handler
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState  |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; 
	oy = y;
    glutPostRedisplay();
}

// GLUT motion handler
void motion(int x, int y)
{
    float dx, dy;
    dx = x - ox;
    dy = y - oy;

    if (buttonState == 3) {
        // left + middle = zoom
        viewTranslation.z += dy / 100.0;
    } 
    else if (buttonState & 2) {
        // middle = translate
        viewTranslation.x += dx / 100.0;
        viewTranslation.y -= dy / 100.0;
    }
    else if (buttonState & 1) {
        // left = rotate
        viewRotation.x += dy / 5.0;
        viewRotation.y += dx / 5.0;
    }

    ox = x; 
	oy = y;
    glutPostRedisplay();
}

// initialize GLUT
void initGl(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(dispWidth, dispHeight);
    mainWindow = glutCreateWindow("Direct Curve Volume Render");

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);
    }
}

// Swap between big-endian double and little-endian double
double swap(double d)
{
	union
	{
		double value;
		uchar bytes[];
	} in, out;

	in.value = d;

	for (uint i = 0; i < sizeof(double); i++)
	{
		out.bytes[i] = in.bytes[sizeof(double) - i - 1];
	}
  return out.value;
}

// Read data from an 'unsigned_short' binary VTK file
bool readVtk(const char* filename, int* width, int* height, int* depth, float* meanNorm, ushort** data)
{
	printf("Reading VTK file '%s'\n", filename);
	FILE *fp = fopen(filename, "rb");
	if (fp == NULL)
	{
		return false;
	}
	char line[1000];
	char dataType[1000];
	int w, h, d;
	while(fgets(line, 1000, fp) != NULL)
	{
		if (sscanf(line, "DIMENSIONS %d %d %d", &w, &h, &d))
		{
			*width = w;
			*height = h;
			*depth = d;
		}
		if (sscanf(line, "SCALARS %s %s", dataType, dataType) * w * h * d)
		{
			break;
		}
	}
	if (strlen(dataType) * w * h * d == 0)
	{
		return false;
	}


	if (strcmp(dataType, "unsigned_short") == 0) 
	{
		fseek(fp, -w * h * d * sizeof(ushort) + 1, SEEK_END);

		ushort* rawData = (ushort*) malloc(sizeof(ushort) * w * h * d);
		size_t dataRead = fread(rawData, sizeof(ushort), w * h * d, fp);
		fclose(fp);
		if (dataRead == 0)
		{
			return false;
		}

		double minVal = (double) rawData[0];
		double maxVal = (double) rawData[0];
		for (uint i = 0; i < w * h * d; i++) 
		{
			minVal = min(minVal, (double) rawData[i]);
			maxVal = max(maxVal, (double) rawData[i]);
		}

		float meanNormVal = 0.1f;
		*data = (ushort*) malloc(sizeof(ushort) * w * h * d);
		for (uint i = 0; i < w * h * d; i++) 
		{
			float val = (float) ((((double) rawData[i]) - minVal) / (maxVal - minVal));
			meanNormVal += val;
			(*data)[i] = (ushort) (USHRT_MAX * val);
		}
		free(rawData);
		meanNormVal /= w * h * d;
		*meanNorm = meanNormVal;


		return meanNormVal > 0;
	}
	if (strcmp(dataType, "double") == 0) 
	{
		fseek(fp, -w * h * d * sizeof(double), SEEK_END);
		double* rawData = (double*) malloc(sizeof(double) * w * h * d);
		size_t dataRead = fread(rawData, sizeof(double), w * h * d, fp);
		fclose(fp);
		if (dataRead == 0)
		{
			return false;
		}

		double minVal = swap(rawData[0]);
		double maxVal = swap(rawData[0]);
		for (uint i = 0; i < w * h * d; i++) 
		{
			minVal = min(minVal, swap(rawData[i]));
			maxVal = max(maxVal, swap(rawData[i]));
		}

		float meanNormVal = 0.1f;
		*data = (ushort*) malloc(sizeof(ushort) * w * h * d);
		for (uint i = 0; i < w * h * d; i++) 
		{
			float val = (float) ((swap(rawData[i]) - minVal) / (maxVal - minVal));
			meanNormVal += val;
			(*data)[i] = (ushort) (USHRT_MAX * val);
		}
		free(rawData);
		meanNormVal /= w * h * d;
		*meanNorm = meanNormVal;

		return meanNormVal > 0;
	}	
	return false;
}

int main(int argc, char* argv[]) 
{
	// Read VTK file
	int width, height, depth;
	float meanNorm;
	ushort* data;
	if (!readVtk(argv[1], &width, &height, &depth, &meanNorm, &data))
	{
		fprintf(stderr, "Could not read VTK file '%s'", argv[1]);
        exit(-1);
	}

	// Get geometry type
	char geomType;
	printf("Data geometry (cylindrical: 0, spherical: 1): ");
	scanf("%d", &geomType);

	// Get geometry size
	float3 geomSize;
	switch(geomType)
	{
	case 0:
		printf("Angle: ");
		scanf("%f", &geomSize.y);
		geomSize.x = width / fmaxf(width, depth);
		geomSize.y = degToRad(geomSize.y);
		geomSize.z = depth / fmaxf(width, depth);
		break;
	case 1:
		printf("Longitude: ");
		scanf("%f", &geomSize.y);
		geomSize.x = 1;
		geomSize.y = degToRad(geomSize.y);
		geomSize.z = geomSize.y * height / (2.0f * width);
		break;
	default : 
		fprintf(stderr, "Wrong geometry");
        exit(-1);
	}

	// Parameters default values
	dispWidth = 512;
	dispHeight = 512;
	viewTranslation = make_float3(0.0, 0.0, -4.0f);
	blockSize = dim3(16, 16);
	triCubic = 1;
	rayStep = 0.001;
	compoType = 0;
	density = 0.05f;
	regBrightness = 1.0f;
	isoValue = 0.0f;
	transferScale = 1.0f / (2.0f * meanNorm);
	xrayBrightness = 1.0f / (2.0f * meanNorm);
	float4 transferFunc[9] = {{0.0, 0.0, 0.0, 0.0},
							  {1.0, 0.0, 0.0, 1.0},
							  {1.0, 0.5, 0.0, 1.0},
							  {1.0, 1.0, 0.0, 1.0},
							  {0.0, 1.0, 0.0, 1.0},
							  {0.0, 1.0, 1.0, 1.0},
					          {0.0, 0.0, 1.0, 1.0},
							  {0.0, 0.0, 1.0, 1.0},
							  {0.0, 0.0, 0.0, 0.0}};

	// Initiate GLUT
	initGl(argc, argv);

	// Initiate device
    cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());

	// Load geometry information
	loadGeometry(&geomSize, &geomType);

	// Load data
	cudaExtent dataDim = make_cudaExtent(width, height, depth);
	loadData(data, &dataDim);
	free(data);

	// Load transfer function
	loadTransferFunc(transferFunc);

	// Initiate timer
	cutilCheckError( cutCreateTimer(&timer) );

	// Set GLUT handlers
	glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

	// Initiate PBO
    initPixelBuffer();
	
	// Display GUI
	GLUI* glui = GLUI_Master.create_glui("Parameters");

	glui->add_checkbox("Tri-cubic interpolation", &triCubic);

	GLUI_Spinner* stepSpinner = glui->add_spinner("Ray step:", GLUI_SPINNER_FLOAT, &rayStep);
	stepSpinner->set_float_limits(0.0001f, 0.1f);
	stepSpinner->set_float_val(rayStep);

	GLUI_RadioGroup* compositionRadio = glui->add_radiogroup(&compoType);
	glui->add_radiobutton_to_group(compositionRadio,"Regular composition");
	glui->add_radiobutton_to_group(compositionRadio,"X-ray");

	GLUI_Spinner* isoValueSpinner = glui->add_spinner("ISO value:", GLUI_SPINNER_FLOAT, &isoValue);
	isoValueSpinner->set_float_limits(0.0f, 1.0f);
	isoValueSpinner->set_float_val(isoValue);

	GLUI_Panel* regularCompoPanel = glui->add_panel("Regular composition parameters");

	GLUI_Spinner* regBrightnessSpinner = glui->add_spinner_to_panel(regularCompoPanel, "Brightness:", GLUI_SPINNER_FLOAT, &regBrightness);
	regBrightnessSpinner->set_float_limits(0.0f, 100.0f);
	regBrightnessSpinner->set_float_val(regBrightness);

	GLUI_Spinner* densitySpinner = glui->add_spinner_to_panel(regularCompoPanel, "Density:", GLUI_SPINNER_FLOAT, &density);
	densitySpinner->set_float_limits(0.0f, 0.1f);
	densitySpinner->set_float_val(density);

	GLUI_Spinner* scaleSpinner = glui->add_spinner_to_panel(regularCompoPanel, "Transfer scale:", GLUI_SPINNER_FLOAT, &transferScale);
	scaleSpinner->set_float_limits(0.0f, 100.0f);
	scaleSpinner->set_float_val(transferScale);

	GLUI_Panel* xrayCompoPanel = glui->add_panel("X-ray parameters");

	GLUI_Spinner* xrayBrightnessSpinner = glui->add_spinner_to_panel(xrayCompoPanel, "Brightness:", GLUI_SPINNER_FLOAT, &xrayBrightness);
	xrayBrightnessSpinner->set_float_limits(0.0f, 100.0f);
	xrayBrightnessSpinner->set_float_val(xrayBrightness);

	glui->set_main_gfx_window(mainWindow);
	GLUI_Master.set_glutIdleFunc(idle);

	// Main loop
    glutMainLoop();

	// Exit handlers
    cudaThreadExit();
    atexit(cleanup);

	return 0;
}