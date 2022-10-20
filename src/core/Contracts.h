#ifndef Contracts_h
#define Contracts_h

#include <stdio.h>

#ifdef DEBUG
    #define ASSERT(cond, msg)  if (!cond) fprintf(stderr, "Assertion failure on line %d... Msg: %s\n", __LINE__, msg) 
#else
    #define ASSERT(cond, msg) 
#endif

#ifdef DEBUG
    #define REQUIRES(cond)  if (!cond) fprintf(stderr, "Assertion failure on line %d... Msg: %s\n", __LINE__, msg) 
#else
    #define REQUIRES(cond) 
#endif

#ifdef DEBUG
    #define ENSURES(cond)  if (!cond) fprintf(stderr, "Assertion failure on line %d... Msg: %s\n", __LINE__, msg) 
#else
    #define ENSURES(cond) 
#endif

#endif