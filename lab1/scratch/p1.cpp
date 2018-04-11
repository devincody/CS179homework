#include <cstdio>
#include <cstdlib>

int main() {
    int *a = (int *) malloc(sizeof (int));
    scanf("%d", a);
    if (!*a)
        printf("Value is 0\n");
    return 0;
}



