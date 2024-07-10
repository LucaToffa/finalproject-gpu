#pragma once

#if DEBUG
    #define PRINTF(...) printf(__VA_ARGS__)
#else
    #define PRINTF(...)
#endif
#ifndef TILE_SIZE //here just to avoid errors in editor
    #define TILE_SIZE 32 
#endif
#ifndef BLOCK_ROWS
    #define BLOCK_ROWS 8 //works up to 16
    
#endif
#define DEFAULT_SIZE 32
#define TRANSPOSITIONS 100

