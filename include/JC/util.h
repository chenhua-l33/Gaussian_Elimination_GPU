#pragma once

#include <exception>
#include <iostream>
#include <iomanip> // for setw and setfill (used with cout to align output), setprecision
#include <sstream>
#include <string>
#include <vector>
#include <random> // also contains RAND_MAX
#include <cmath>  // log, pow, ...

using namespace std;

bool PRINT_DATA = false; // set this to true to print input & output data. Use this to debug

#ifndef _int64
#define _int64 long long
#endif

// return smallest x >= size that is a multiple of the divisor
unsigned int closestMultiple(unsigned int size, unsigned int divisor)
{
    unsigned int remainder = size % divisor;
    return remainder == 0 ? size : size - remainder + divisor;
}

#ifdef WIN32
#include <direct.h>  // http://www.codebind.com/cprogramming/get-current-directory-using-c-program/  also linux
#define GetCurrentDir _getcwd
#else // for linux:
#include <unistd.h>
#define GetCurrentDir getcwd
#endif
string currentFolder() {
	char buff[FILENAME_MAX];
	GetCurrentDir(buff, FILENAME_MAX);
	return string(buff);
}

// for creating the following sequence: 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
// for-loop:  for (int i = 1; i < 10000; i += delta(i))
int delta(int x, int slowdown) {
	if (x <= slowdown)
		return 1;
	x /= slowdown;
	// make it a power of 2
	x = (int)(log(x) / log(2));
	return (int)pow(2, x);
}
int delta(int x) {
	return delta(x, 2);
}

template <class T>
void showMatrix(T *matrix, unsigned int width, unsigned int height)
{
    for (unsigned int row = 0; row < height; ++row) {
        for (unsigned int col = 0; col < width; ++col) {
            cout << matrix[width*row + col] << " ";
        }
        cout << endl;
    }
    return;
}

template <class T>
void printArray(T ARRAY[], int size) {
	cout << "[";
	for (int i = 0; i < size; i++) {
		if (i > 0) cout << ", ";
		cout << setprecision(3) << ARRAY[i];
	}
	cout << "]" << endl;
}
float sumArray(float ARRAY[], int size) {
	float sum = 0;
	for (int i = 0; i < size; i++)
		sum += ARRAY[i];
	return sum;
}
// MAX_NUMBER: exclusive
template <class T>
T* initializeArray(const int size, const T MIN_NUMBER, const T MAX_NUMBER) {
	srand(time(NULL)); // random initial seed

	T* array = new T[size];
	double SCALE = (double)(MAX_NUMBER - MIN_NUMBER) / RAND_MAX;
	for (int i = 0; i < size; i++) {
//#if INT_MAX == RAND_MAX  // on linux (RAND_MAX + 1 would result in overflow for integers)
	//	cout << "INT_MAX == RAND_MAX" << endl;
		array[i] = (T)(rand() * SCALE + MIN_NUMBER);
//#else   // on windows is RAND_MAX = 2^15-1 = 32767
	//	cout << "INT_MAX != RAND_MAX" << endl;
	//	array[i] = rand() * (MAX_NUMBER - MIN_NUMBER) / (RAND_MAX + 1) + MIN_NUMBER;
//#endif
	}
	return array;
}

//typedef float TYPE; // 
//
//// allocate and initialize array with random variables
//TYPE* initializeArray(const int size, const TYPE MIN_NUMBER, const TYPE MAX_NUMBER) {
//	srand(time(NULL)); // random initial seed
//
//	TYPE* array = new TYPE[size];
//	TYPE SCALE = (MAX_NUMBER - MIN_NUMBER) / (RAND_MAX + 1);
//	for (int i = 0; i < size; i++)
//		array[i] = rand() * SCALE + MIN_NUMBER;
//	return array;
//}


// *** utility functions ***
#ifndef FLT_MIN
#define FLT_MIN          2e-38F
#endif
bool closeEnough(float a, float b, float resolution)
{

	float abs_a = abs(a);
	float abs_b = abs(b);
	float delta = abs(a - b);

	if (a == b) {
		return true;
	}

	if (a == 0 || b == 0 || delta < FLT_MIN) {
		return delta < resolution;
	}

	return delta / (abs_a + abs_b) < resolution;
}

template <class T>
bool checkIfResultsAreTheSame(T* arr3, T* arr4, int n, bool PRINT) {
	int nbr_diff = 0;
	for (int i = 0; i < n; i++) {
		if (arr3[i] != arr4[i]) {
			nbr_diff++;
			if (PRINT && nbr_diff < 10)
				cout <<"["<<i<<"] "<< arr3[i] << "<>" << arr4[i] << endl;
		}
	}
	if (nbr_diff > 0)
		cout << "Results CPU-GPU are not the same: " << nbr_diff << " differences!!!!!!" << endl;
	else
		cout << "Results of CPU and GPU are the same. OK. " << endl;
	return nbr_diff == 0;
}

// use this one for floating point types
template <class T>
bool checkIfResultsAreTheSame(T* arr3, T* arr4, int n, T precision, bool PRINT) {
	int nbr_diff = 0;
	for (int i = 0; i < n; i++) {
		T rel_diff = arr3[i] == arr4[i] ? 0 : abs(arr3[i] - arr4[i]) / (arr3[i] == 0 ? arr4[i] : arr3[i]);
		if (rel_diff > precision) {
			nbr_diff++;
			if (PRINT && nbr_diff < 10)
				cout << "[" << i << "] " << arr3[i] << "<>" << arr4[i] << endl;
		}
	}
	if (nbr_diff > 0)
		cout << "Results CPU-GPU are not the same: " << nbr_diff << " differences!!!!!!" << endl;
	else
		cout << "Results of CPU and GPU are the same. OK. " << endl;
	return nbr_diff == 0;
}
// ******** STATISTICS ***********
template <class T> T minimalValue(T* values, int NBR) {
	T min = values[0];
	for (int i = 1; i < NBR; i++) {
		if (values[i] < min)
			min = values[i];
	}
	return min;
}
template <class T> T meanValue(T* values, int NBR) {
	T sum = values[0];
	for (int i = 1; i < NBR; i++) {
		sum += values[i];
	}
	return sum / NBR;
}
template <class T> T stddev(T* values, int NBR) {
	T mean = meanValue(values, NBR);
	T SSE = 0;
	for (int i = 0; i < NBR; i++) {
		SSE += (values[i] - mean) * (values[i] - mean);
	}
	return sqrt(SSE / NBR);
}
// relative standard deviation from the mean
template <class T> float relstddev(T* values, int NBR) {
	T mean = meanValue(values, NBR);
	T SSE = 0;
	for (int i = 0; i < NBR; i++) {
		SSE += (values[i] - mean) * (values[i] - mean);
	}
	float stddev = sqrt(SSE / NBR);
	return stddev / mean;
}
// ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
// ********** Parse Program Arguments argc & argc ************
// ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
int defaultOrViaArgs(int defaultValue, char option, int argc, char* const argv[]) {
	char option_str[3];
	option_str[0] = '-'; option_str[1] = option; option_str[2] = '\0';
	for (int i = 1; i < argc; i++) {
		if (strncmp(argv[i], option_str, 2) == 0) {
			i++;
			if (i < argc) {
				return atoi(argv[i]);
			}
			else {
				ostringstream oss;
				oss << "Option -" << option << " expects an integer argument..." << endl;
				throw runtime_error(oss.str());
			}
		}
	}
	return defaultValue;
}

bool argsContainsOption(char option, int argc, char* const argv[]) {
	char option_str[3];
	option_str[0] = '-'; option_str[1] = option; option_str[2] = '\0';
	for (int i = 1; i < argc; i++) 
		if (strncmp(argv[i], option_str, 2) == 0)
			return true;
	return false;
}

bool argsContainsUnknownOption(const char *options, int argc, char* const argv[]) {
	//cout<< "Illegal option -- %c\n", optopt);
	for (int i = 1; i < argc; i++) {
		if (argv[i][0] == '-') {
			const char* ch_ptr = strchr(options, argv[i][1]);
			if (ch_ptr == NULL) {
				cout << "Illegal option: '"<< argv[i][1] <<"'. Allowed options: "<<options<<"."<<endl;
				return true;
			}
		}
	}
	return false;
}

// ALTERNATIVE (USED A LOT)
////getopt() is actually a really simple function.I made a github gist for it, code from here is below too
//// from https://stackoverflow.com/questions/10404448/getopt-h-compiling-linux-c-code-in-windows
//#include <string.h>
//#include <stdio.h>
//
//int     opterr = 1,             /* if error message should be printed */
//optind = 1,             /* index into parent argv vector */
//optopt,                 /* character checked for validity */
//optreset;               /* reset getopt */
//char    *optarg;                /* argument associated with option */
//
//#define BADCH   (int)'?'
//#define BADARG  (int)':'
//#define EMSG    ""
//
///*
//* getopt --
//*      Parse argc/argv argument vector.
//*/
//int getopt(int nargc, char * const nargv[], const char *ostr)
//{
//	static char *place = EMSG;              /* option letter processing */
//	const char *oli;                        /* option letter list index */
//
//	if (optreset || !*place) {              /* update scanning pointer */
//		optreset = 0;
//		if (optind >= nargc || *(place = nargv[optind]) != '-') {
//			place = EMSG;
//			return (-1);
//		}
//		if (place[1] && *++place == '-') {      /* found "--" */
//			++optind;
//			place = EMSG;
//			return (-1);
//		}
//	}                                       /* option letter okay? */
//	if ((optopt = (int)*place++) == (int)':' ||
//		!(oli = strchr(ostr, optopt))) {
//		/*
//		* if the user didn't specify '-' as an option,
//		* assume it means -1.
//		*/
//		if (optopt == (int)'-')
//			return (-1);
//		if (!*place)
//			++optind;
//		if (opterr && *ostr != ':')
//			(void)printf("illegal option -- %c\n", optopt);
//		return (BADCH);
//	}
//	if (*++oli != ':') {                    /* don't need argument */
//		optarg = NULL;
//		if (!*place)
//			++optind;
//	}
//	else {                                  /* need an argument */
//		if (*place)                     /* no white space */
//			optarg = place;
//		else if (nargc <= ++optind) {   /* no arg */
//			place = EMSG;
//			if (*ostr == ':')
//				return (BADARG);
//			if (opterr)
//				(void)printf("option requires an argument -- %c\n", optopt);
//			return (BADCH);
//		}
//		else                            /* white space */
//			optarg = nargv[optind];
//		place = EMSG;
//		++optind;
//	}
//	return (optopt);                        /* dump back option letter */
//}

