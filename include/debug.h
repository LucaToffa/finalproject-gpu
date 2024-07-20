#ifndef __DEBUG_H__
#define __DEBUG_H__

#ifdef DEBUG
    #define PRINTF(...) printf(__VA_ARGS__)
#else
    #define PRINTF(...)
#endif

#endif
