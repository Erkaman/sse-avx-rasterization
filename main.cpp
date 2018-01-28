/*
The MIT License (MIT)
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*

This code is entirely self-contained, and depends on no libraries.
Running the program will rasterize a single triangle, and write it to `out.ppm`
You can convert this file into a more common format like png using something like imagemagick.
This image `out.ppm` should look exactly like the provided image triangle.png

Note that the code is very simple, and does no out-of-bounds error checking.

Note that this file may only compile in Visual Studio. For other compilers, you might have to
use different headers to access AVX and SSE, and replace `_aligned_malloc` with something else.

*/

// this macro controls rasterization mode.
// if 0, then normal(no vector instructions),
// if 0, then SSE
// if 2, then AVX
#define RASTERIZE_MODE 0

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>

// get SSE and AVX
#include <intrin.h>

struct vec2 {
public:
	float x;
	float y;

	vec2() : x(0.0f), y(0.0f) { }
	vec2(const float x_, const float y_) : x(x_), y(y_) { }
};

unsigned int rounddownAligned(unsigned int i, unsigned int align) {
	return (unsigned int)floor((float)i / (float)align) * align;
}

unsigned int roundupAligned(unsigned int i, unsigned int align) {
	return (unsigned int)ceil((float)i / (float)align) * align;
}

float clamp(float f, float min, float max) {
	if (f < min) {
		return min;
	}
	else if (f > max) {
		return max;
	}
	else {
		return f;
	}
}

float min(float x, float y) {
	return x < y ? x : y;
}

float max(float x, float y) {
	return x > y ? x : y;
}

__m128 edgeFunctionSSE(const vec2 &a, const vec2 &b, __m128 cx, __m128 cy)
{
	return _mm_sub_ps(_mm_mul_ps(_mm_sub_ps(cx, _mm_set1_ps(a.x)), _mm_set1_ps(b.y - a.y)), _mm_mul_ps(_mm_sub_ps(cy, _mm_set1_ps(a.y)), _mm_set1_ps(b.x - a.x)));
}

__m256 edgeFunctionAVX(const vec2 &a, const vec2 &b, __m256 cx, __m256 cy)
{
	return _mm256_sub_ps(_mm256_mul_ps(_mm256_sub_ps(cx, _mm256_set1_ps(a.x)), _mm256_set1_ps(b.y - a.y)), _mm256_mul_ps(_mm256_sub_ps(cy, _mm256_set1_ps(a.y)), _mm256_set1_ps(b.x - a.x)));
}

float edgeFunction(const vec2 &a, const vec2 &b, const vec2 &c)
{
	// we are doing the reversed edge test, compared to the article.
	// we need to do it in this way, since our coordinate system has the origin in the top-left corner.
	return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

void rasterizeTriangle(
	const float x0, const float y0,
	const float x1, const float y1,
	const float x2, const float y2,
	const unsigned int fbWidth, const unsigned int fbHeight,
	unsigned int* framebuffer
) {
	vec2 vcoords[3];
	vcoords[0] = vec2(x0, y0);
	vcoords[1] = vec2(x1, y1);
	vcoords[2] = vec2(x2, y2);

	// min of triangle AABB
	vec2 amin(+FLT_MAX, +FLT_MAX);
	// max of triangle AABB
	vec2 amax(-FLT_MAX, -FLT_MAX);

	// find triangle AABB.
	{
		for (int i = 0; i < 3; ++i) {
			vec2 p = vcoords[i];

			amin.x = min(p.x, amin.x);
			amin.y = min(p.y, amin.y);

			amax.x = max(p.x, amax.x);
			amax.y = max(p.y, amax.y);
		}

		// make sure we don't rasterize outside the framebuffer..
		amin.x = clamp(amin.x, -1.0f, +1.0f);
		amax.x = clamp(amax.x, -1.0f, +1.0f);

		amin.y = clamp(amin.y, -1.0f, +1.0f);
		amax.y = clamp(amax.y, -1.0f, +1.0f);
	}

	const float intAminx = float(rounddownAligned(unsigned int((0.5f + 0.5f * amin.x)* fbWidth), 1));
	const float intAmaxx = float(roundupAligned(unsigned int((0.5f + 0.5f * amax.x)* fbWidth), 1));
	const float intAminy = float((0.5f + 0.5f * amin.y)* fbHeight);
	const float intAmaxy = float((0.5f + 0.5f * amax.y)* fbHeight);

	const float doublePixelWidth = 2.0f / (float)fbWidth;
	const float doublePixelHeight = 2.0f / (float)fbHeight;

	vec2 p;

	for (float iy = intAminy; iy <= (float)intAmaxy; iy += 1.0f) {
		// map from [0,height] to [-1,+1]
		p.y = -1.0f + iy * doublePixelHeight;
		for (float ix = intAminx; ix <= (float)intAmaxx; ix += 1.0f) {
			// map from [0,width] to [-1,+1]
			p.x = -1.0f + ix * doublePixelWidth;

			float w0 = edgeFunction(vcoords[1], vcoords[2], p);
			float w1 = edgeFunction(vcoords[2], vcoords[0], p);
			float w2 = edgeFunction(vcoords[0], vcoords[1], p);

			// is it on the right side of all edges?
			if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) {

				unsigned int iBuf = (unsigned int)(iy * fbWidth + ix);
				framebuffer[iBuf] = 0xFFFFFFFF;
			}
		}
	}
}

void rasterizeTriangleSSE(
	const float x0, const float y0,
	const float x1, const float y1,
	const float x2, const float y2,
	const unsigned int fbWidth,
	const unsigned int fbHeight,
	unsigned int* framebuffer) {
	vec2 vcoords[3];
	vcoords[0] = vec2(x0, y0);
	vcoords[1] = vec2(x1, y1);
	vcoords[2] = vec2(x2, y2);

	// min of triangle AABB
	vec2 amin(+FLT_MAX, +FLT_MAX);
	// max of triangle AABB
	vec2 amax(-FLT_MAX, -FLT_MAX);

	// find triangle AABB.
	{
		for (int i = 0; i < 3; ++i) {
			vec2 p = vcoords[i];

			amin.x = min(p.x, amin.x);
			amin.y = min(p.y, amin.y);

			amax.x = max(p.x, amax.x);
			amax.y = max(p.y, amax.y);
		}

		// make sure we don't rasterize outside the framebuffer..
		amin.x = clamp(amin.x, -1.0f, +1.0f);
		amax.x = clamp(amax.x, -1.0f, +1.0f);

		amin.y = clamp(amin.y, -1.0f, +1.0f);
		amax.y = clamp(amax.y, -1.0f, +1.0f);
	}

	// float where are all bits are set.
	float filledbitsfloat;
	{
		unsigned int ii = 0xffffffff;
		memcpy(&filledbitsfloat, &ii, sizeof(float));
	}
	float whitecolorfloat = filledbitsfloat;

	/*
	We'll be looping over all pixels in the AABB, and rasterize the pixels within the triangle. The AABB has been
	extruded on the x-axis, and aligned to 16bytes.
	This is necessary since _mm_store_ps can only write to 16-byte aligned addresses.
	*/
	const float intAminx = (float)rounddownAligned(int((0.5f + 0.5f * amin.x)* fbWidth), 4); // extrude
	const float intAmaxx = (float)roundupAligned(int((0.5f + 0.5f * amax.x)* fbWidth), 4); // extrude
	const float intAminy = (float)int((0.5f + 0.5f * amin.y)* fbHeight);
	const float intAmaxy = (float)int((0.5f + 0.5f * amax.y)* fbHeight);

	const float doublePixelWidth = 2.0f / (float)fbWidth;
	const float doublePixelHeight = 2.0f / (float)fbHeight;

	__m128 minusone = _mm_set1_ps(-1.0f);
	__m128 zero = _mm_setzero_ps();

	for (float iy = intAminy; iy <= (float)intAmaxy; iy += 1.0f) {
		// map from [0,height] to [-1,+1]
		__m128 py = _mm_add_ps(minusone, _mm_mul_ps(_mm_set1_ps(iy), _mm_set1_ps(doublePixelHeight)));

		for (float ix = intAminx; ix <= (float)intAmaxx; ix += 4.0f) {
			// this `px` register contains the x-coords of four pixels in a row.
			// we map from [0,width] to [-1,+1]
			__m128 px = _mm_add_ps(minusone, _mm_mul_ps(
				_mm_set_ps(ix + 3.0f, ix + 2.0f, ix + 1.0f, ix + 0.0f), _mm_set1_ps(doublePixelWidth)));

			__m128 w0 = edgeFunctionSSE(vcoords[1], vcoords[2], px, py);
			__m128 w1 = edgeFunctionSSE(vcoords[2], vcoords[0], px, py);
			__m128 w2 = edgeFunctionSSE(vcoords[0], vcoords[1], px, py);

			// the default bitflag, results in all the four pixels being overwritten.
			__m128 writeFlag = _mm_set_ps1(filledbitsfloat);

			// the results of the edge tests are used to modify our bitflag.
			writeFlag = _mm_and_ps(writeFlag, _mm_cmpge_ps(w0, zero));
			writeFlag = _mm_and_ps(writeFlag, _mm_cmpge_ps(w1, zero));
			writeFlag = _mm_and_ps(writeFlag, _mm_cmpge_ps(w2, zero));

			unsigned int iBuf = unsigned int(iy * fbWidth + ix);

			__m128 newBufferVal = _mm_set1_ps(whitecolorfloat);
			__m128 origBufferVal = _mm_load_ps((const float*)framebuffer + iBuf);

			/*
			We only want to write to pixels that are inside the triangle.
			However, implementing such a conditional write is tricky when dealing SIMD.

			We implement this by using a bitflag. This bitflag determines which of the four floats in __m128 should
			just write the old value to the buffer(meaning that the pixel is NOT actually rasterized),
			and which should overwrite the current value in the buffer(meaning that the pixel IS rasterized).

			This is implemented by some bitwise manipulation tricks.
			*/
			_mm_store_ps((float*)(framebuffer)+iBuf,
				_mm_or_ps(
					_mm_and_ps(writeFlag, newBufferVal),
					_mm_andnot_ps(writeFlag, origBufferVal)
				));
		}
	}
}

void rasterizeTriangleAVX(
	const float x0, const float y0,
	const float x1, const float y1,
	const float x2, const float y2,
	const unsigned int fbWidth,
	const unsigned int fbHeight,
	unsigned int* framebuffer) {
	vec2 vcoords[3];
	vcoords[0] = vec2(x0, y0);
	vcoords[1] = vec2(x1, y1);
	vcoords[2] = vec2(x2, y2);

	// min of triangle AABB
	vec2 amin(+FLT_MAX, +FLT_MAX);
	// max of triangle AABB
	vec2 amax(-FLT_MAX, -FLT_MAX);

	// find triangle AABB.
	{
		for (int i = 0; i < 3; ++i) {
			vec2 p = vcoords[i];

			amin.x = min(p.x, amin.x);
			amin.y = min(p.y, amin.y);

			amax.x = max(p.x, amax.x);
			amax.y = max(p.y, amax.y);
		}

		// make sure we don't rasterize outside the framebuffer..
		amin.x = clamp(amin.x, -1.0f, +1.0f);
		amax.x = clamp(amax.x, -1.0f, +1.0f);

		amin.y = clamp(amin.y, -1.0f, +1.0f);
		amax.y = clamp(amax.y, -1.0f, +1.0f);
	}

	// float where are all bits are set.
	float filledbitsfloat;
	{
		unsigned int ii = 0xffffffff;
		memcpy(&filledbitsfloat, &ii, sizeof(float));
	}
	float whitecolorfloat = filledbitsfloat;

	/*
	We'll be looping over all pixels in the AABB, and rasterize the pixels within the triangle. The AABB has been
	extruded on the x-axis, and aligned to 16bytes.
	This is necessary since _mm_store_ps can only write to 16-byte aligned addresses.
	*/
	const float intAminx = (float)rounddownAligned(int((0.5f + 0.5f * amin.x)* fbWidth), 8); // extrude
	const float intAmaxx = (float)roundupAligned(int((0.5f + 0.5f * amax.x)* fbWidth), 8); // extrude
	const float intAminy = (float)((0.5f + 0.5f * amin.y)* fbHeight);
	const float intAmaxy = (float)((0.5f + 0.5f * amax.y)* fbHeight);

	const float doublePixelWidth = 2.0f / (float)fbWidth;
	const float doublePixelHeight = 2.0f / (float)fbHeight;

	__m256 minusone = _mm256_set1_ps(-1.0f);
	__m256 zero = _mm256_setzero_ps();

	for (float iy = intAminy; iy <= intAmaxy; iy += 1.0f) {
		// map from [0,height] to [-1,+1]
		__m256 py = _mm256_add_ps(minusone, _mm256_mul_ps(_mm256_set1_ps(iy), _mm256_set1_ps(doublePixelHeight)));

		for (float ix = intAminx; ix <= intAmaxx; ix += 8.0f) {
			// this `px` register contains the x-coords of four pixels in a row.
			// we map from [0,width] to [-1,+1]
			__m256 px = _mm256_add_ps(minusone, _mm256_mul_ps(
				_mm256_set_ps(ix + 7.0f, ix + 6.0f, ix + 5.0f, ix + 4.0f, ix + 3.0f, ix + 2.0f, ix + 1.0f, ix + 0.0f), _mm256_set1_ps(doublePixelWidth)));

			__m256 w0 = edgeFunctionAVX(vcoords[1], vcoords[2], px, py);
			__m256 w1 = edgeFunctionAVX(vcoords[2], vcoords[0], px, py);
			__m256 w2 = edgeFunctionAVX(vcoords[0], vcoords[1], px, py);

			// the default bitflag, results in all the four pixels being overwritten.
			__m256 writeFlag = _mm256_set1_ps(filledbitsfloat);

			// the results of the edge tests are used to modify our bitflag.

			writeFlag = _mm256_and_ps(writeFlag, _mm256_cmp_ps(w0, zero, _CMP_NLT_US));
			writeFlag = _mm256_and_ps(writeFlag, _mm256_cmp_ps(w1, zero, _CMP_NLT_US));
			writeFlag = _mm256_and_ps(writeFlag, _mm256_cmp_ps(w2, zero, _CMP_NLT_US));


			//writeFlag = _mm256_and_ps(writeFlag, _mm256_cmp_ps(w0, zero, _CMP_NLT_US));


			unsigned int iBuf = unsigned int(iy * fbWidth + ix);

			__m256 newBufferVal = _mm256_set1_ps(whitecolorfloat);
			__m256 origBufferVal = _mm256_load_ps((const float*)framebuffer + iBuf);

			/*
			We only want to write to pixels that are inside the triangle.
			However, implementing such a conditional write is tricky when dealing SIMD.

			We implement this by using a bitflag. This bitflag determines which of the four floats in __m128 should
			just write the old value to the buffer(meaning that the pixel is NOT actually rasterized),
			and which should overwrite the current value in the buffer(meaning that the pixel IS rasterized).

			This is implemented by some bitwise manipulation tricks.
			*/
			_mm256_store_ps((float*)(framebuffer)+iBuf,
				_mm256_or_ps(
					_mm256_and_ps(writeFlag, newBufferVal),
					_mm256_andnot_ps(writeFlag, origBufferVal)
				));
		}
	}
}


int main() {
	const unsigned int WIDTH = 480;
	const unsigned int HEIGHT = 360;

	unsigned int* framebuffer = (unsigned int*)_aligned_malloc(WIDTH*HEIGHT * sizeof(unsigned int), sizeof(unsigned int) * 8);

	// clear framebuffer
	for (size_t row = 0; row < HEIGHT; ++row) {
		for (size_t col = 0; col < WIDTH; ++col) {
			framebuffer[row * WIDTH + col] = (255 << 0) | (0 << 8) | (0 << 16);
		}
	}

	// triangle vertices. in range [-1.0, +1.0]
	float x0 = 0.1f; float y0 = 0.65f;
	float x1 = +0.75f; float y1 = -0.85f;
	float x2 = -0.60f; float y2 = -0.65f; 
	
	if (RASTERIZE_MODE == 0) {
		printf("rasterize normal\n");
		rasterizeTriangle(x0, y0, x1, y1, x2, y2, WIDTH, HEIGHT, framebuffer);
	} else if (RASTERIZE_MODE == 1) {
		printf("rasterize SSE\n");
		rasterizeTriangleSSE(x0, y0,x1, y1, x2, y2, WIDTH, HEIGHT, framebuffer);
	}
	else {
		printf("rasterize AVX\n");
		rasterizeTriangleAVX(x0, y0, x1, y1, x2, y2, WIDTH, HEIGHT, framebuffer);
	}
	
	// output the rasterized triangle to `out.ppm`
	{
		FILE* fp = fopen("out.ppm", "w");
		if (fp == nullptr) {
			printf("could not open output file for writing!\n");
			return 1;
		}

		fprintf(fp, "P3\n");

		fprintf(fp, "%d %d\n", WIDTH, HEIGHT);
		fprintf(fp, "255\n");

		for (unsigned int row = 0; row < HEIGHT; ++row) {

			for (unsigned int col = 0; col < WIDTH; ++col) {

				unsigned int c = framebuffer[row * WIDTH + col];

				unsigned char r = 0xFF & (c >> 0);
				unsigned char g = 0xFF & (c >> 8);
				unsigned char b = 0xFF & (c >> 16);

				fprintf(fp, "%d %d %d ", r, g, b);

			}
			fprintf(fp, "\n");
		}

		fclose(fp);
	}

	_aligned_free(framebuffer);
}