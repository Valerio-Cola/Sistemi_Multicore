/*
Per ogni pixel allocato in una matrice 2D: red*0.21 + green*0.72 + blue*0.07

La matrice può essere vista come un array 1D composto dalle righe della matrice messe in sequenza

Accesso a singolo pixel [riga*len_riga+colonna]

L'immagine è divisa in blocchi di thread, 1 thread = 1 pixel
Può essere che i blocchi abbiano righe/colonne di pixel che non devono lavorare
    poichè la dimensione dell'immagine non è precisa alla griglia di blocchi 
*/

//Conversione immagine in bianco e nero
__global__ void colorToGrey(unsigned char *imageColor, unsigned char *imageGrey, int width, int height) {
    // Calcola la posizione globale del pixel 
    int col = blockIdx.x + threadIdx.x * blockDim.x;
    int row = blockIdx.y + threadIdx.y * blockDim.y;

    // Controlla che il pixel sia all'interno dell'immagine
    if(col < width && row < height) {
        // Calcola l'indice del pixel da scrivere
        int index = row * width + col;

        // Moltiplico per il numero di canali di ingresso RGB
        // così da ottenere l'indice del pixel nell'immagine a colori
        int colorIndex = index * 3;

        // Legge i valori dei canali di colore
        unsigned char r = imageColor[colorIndex];
        unsigned char g = imageColor[colorIndex + 1];
        unsigned char b = imageColor[colorIndex + 2];

        // Calcola il valore del pixel in scala di grigio
        imageGrey[index] = (0.21f*r + 0.71f*g + 0.07f*b);
    }
}
#define BLUR_SIZE 3

// Effetto blur sull'immagine, si ottiene applicando al pixel il valore medio di tutti i suoi pixel adiacenti
// Non del tutto corretto poichè non considera i pixel ai bordi di ogni blocco
// Pongo immagine con 1 canale
__global__ void blurImage(unsigned char *imageIn, unsigned char *imageOut, int width, int height) {
    
    // Calcola la posizione globale del pixel
    int col = blockIdx.x + threadIdx.x * blockDim.x;
    int row = blockIdx.y + threadIdx.y * blockDim.y;

    // Controlla che il pixel sia all'interno dell'immagine
    if(col < width && row < height) {
        int pixVal = 0;
        int pixels = 0;

        // Calcola la media dei pixel vicini quindi gli 8 pixel adiacenti per ottenere effetto blur
        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                
                // Indice di ogni pixel vicino
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // Controlla che il pixel sia all'interno dell'immagine
                if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                    pixVal += imageIn[curRow * width + curCol];
                    pixels++;
                }
            }
        }
    // Applica il valore medio calcolato diviso il numero di pixel vicini
    imageOut[row * width + col] = (unsigned char)(pixVal / pixels);
    }
}
