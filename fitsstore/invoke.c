#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv) {
  execv("/astro/iraf/x86_64/gempylocal/bin/python", argv);
  return 0;
}


