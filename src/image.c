#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h> // For uint8_t and uint32_t
#include <ctype.h> // For isdigit

// Function to load a PGM image with unsigned int data
int loadPGM(const char *filename, uint32_t** data, unsigned int* width, unsigned int* height) {
    FILE *fp;
    char magicNumber[3];
    int w, h, mv, i, j;
    char line[256];

    // Open the file in binary read mode
    fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return 0; // Indicate failure
    }

    // Read the magic number
    if (fgets(magicNumber, sizeof(magicNumber), fp) == NULL) {
        fprintf(stderr, "Error: Could not read magic number.\n");
        fclose(fp);
        return 0; // Indicate failure
    }
    magicNumber[strcspn(magicNumber, "\n")] = 0; // Remove trailing newline
    if (strcmp(magicNumber, "P2") != 0 && strcmp(magicNumber, "P5") != 0) {
        fprintf(stderr, "Error: Invalid PGM format (Magic Number: %s)\n", magicNumber);
        fclose(fp);
        return 0; // Indicate failure
    }

    // Skip comments and combine width and height across lines
    w = 0;
    h = 0;
    while (w == 0 || h == 0) {
        if (fgets(line, sizeof(line), fp) == NULL) {
            fprintf(stderr, "Error: Reached end of file while searching for width and height\n");
            fclose(fp);
            return 0;
        }
        if (line[0] == '#') continue;  // Skip comments

        // Try to parse width and height from the line
        if (w == 0 && sscanf(line, "%d %*d", &w) == 1) {
            // Width found, look for height next
            char* remaining = strchr(line, ' ');
            if (remaining != NULL) {
                if (sscanf(remaining, "%d", &h) == 1) {
                    // Both width and height found on the same line
                    break;
                }
            }

        } else if (h == 0 && sscanf(line, "%*d %d", &h) == 1) {
             // Height found, but need width

        }  else if (w == 0 && h == 0 && isdigit(line[0])) {
             // No width or height found, entire line might contain values
              if (sscanf(line, "%d %d", &w, &h) == 2) {
                // Both width and height found
                break;
             }
        }
    }// end while

    if (w == 0 || h == 0) {
        fprintf(stderr, "Error: Could not find width and height after skipping comments\n");
        fclose(fp);
        return 0;
    }

    // Read maxval
    if (fgets(line, sizeof(line), fp) == NULL) {
        fprintf(stderr, "Error: Could not read maxval.\n");
        fclose(fp);
        return 0; // Indicate failure
    }
    if (sscanf(line, "%d", &mv) != 1) {
        fprintf(stderr, "Error: Invalid maxval format.\n");
        fclose(fp);
        return 0; // Indicate failure
    }

    // Allocate memory for the image data (uint32_t)
    *data = (uint32_t *)malloc(w * h * sizeof(uint32_t));
    if (*data == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for pixel data.\n");
        fclose(fp);
        return 0; // Indicate failure
    }

    // Read pixel data based on the format
    if (strcmp(magicNumber, "P2") == 0) { // ASCII PGM
        for (i = 0; i < h; i++) {
            for (j = 0; j < w; j++) {
                unsigned int pixelValue; // Use unsigned int to match uint32_t
                if (fscanf(fp, "%u", &pixelValue) != 1) {  // Use %u for unsigned int
                    fprintf(stderr, "Error: Invalid pixel data format (ASCII).\n");
                    free(*data);
                    *data = NULL;
                    fclose(fp);
                    return 0; // Indicate failure
                }
                (*data)[i * w + j] = (uint32_t)pixelValue; // Store as uint32_t
            }
        }
    } else { // Binary PGM (P5)
        // For uint32_t, need to read as bytes then convert
        uint8_t* byte_data = (uint8_t*)malloc(w * h * sizeof(uint8_t));
        if (byte_data == NULL) {
             fprintf(stderr, "Error: Memory allocation failed for byte data.\n");
             free(*data);
             *data = NULL;
             fclose(fp);
             return 0;
        }

        if (fread(byte_data, sizeof(uint8_t), w * h, fp) != (size_t)(w * h)) {
            fprintf(stderr, "Error: Could not read all pixel data (Binary).\n");
            free(*data);
            free(byte_data);
            *data = NULL;
            fclose(fp);
            return 0; // Indicate failure
        }

        // Convert from byte data to uint32_t data
        for (i = 0; i < w * h; i++) {
            (*data)[i] = (uint32_t)byte_data[i]; // Convert byte to uint32_t
        }
        free(byte_data); // Free temporary byte data
    }

    fclose(fp);

    // Assign values to output parameters only after successful loading
    *width = w;
    *height = h;

    return 1; // Indicate success
}

// Function to save a PGM image with unsigned int data
int savePGM(const char *filename, const uint32_t* data, unsigned int width, unsigned int height) {
    FILE *fp;
    int i;

    // Open the file in write mode
    fp = fopen(filename, "wb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return 0; // Indicate failure
    }

    // Write the header
    fprintf(fp, "P5\n");
    fprintf(fp, "# CREATOR: savePGM function\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");


    uint8_t* byte_data = (uint8_t*)malloc(width * height * sizeof(uint8_t));
    if (byte_data == NULL) {
        fprintf(stderr, "Error allocating byte data for binary write\n");
        fclose(fp);
        return 0;
    }

    //Convert from uint32_t to byte_data

    for(i = 0; i < width * height; i++){
        // if(data[i] > 255){
        //     fprintf(stderr, "Error: uint32_t value %u exceeds byte range\n", data[i]);
        //     free(byte_data);
        //     fclose(fp);
        //     return 0; // Failure: value will be truncated.
        // }
        byte_data[i] = (uint8_t)data[i];
    }


    if (fwrite(byte_data, sizeof(uint8_t), width * height, fp) != (size_t)(width * height)) {
        fprintf(stderr, "Error: Could not write all pixel data (Binary).\n");
        free(byte_data);
        fclose(fp);
        return 0; // Indicate failure
    }
    free(byte_data);

    fclose(fp);
    return 1; // Indicate success
}

