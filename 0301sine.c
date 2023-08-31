#include <stdio.h>

#include <tvmgen_default.h>

static float input;
static float output;
struct tvmgen_default_inputs default_inputs = {.input0 = &input,};
struct tvmgen_default_outputs default_outputs = {.output = &output,};

int main(void)
{
    while(1){
        int ch=0;
        scanf("%d", &ch);
        printf("Input Number is: %d\n", ch);
        
        input = (float)((float)ch*1.0/180.0*3.14159265);

        tvmgen_default_run(&default_inputs, &default_outputs);
        printf("Output Sine Value: %d\n", (int)(output*100));
        
    }
    return 0;
}