# Runing pyTorch Model in STM32F749

>**Model**:\
&emsp;&emsp; SINE MLP-3 Model\
>**Toolchain:** \
&emsp;&emsp; pyTorch + TVM/LLVM + RIOT\
**Procedure:**\
&emsp;&emsp; (1) Use **pyTorch** to train and save model.\
&emsp;&emsp; (2) Use **TVM** to compile model and save it to C library format.\
&emsp;&emsp; (3) Write C file and Makefile to compile the model in **RIOT** OS. \

## Step 1: Train and Save Sine Model in pyTorch

### Training Sine Model
>**File:** `0101trainSine.py`\
**Info:** Refer to [TinyML Book](https://tinymlbook.com/) Chapter 4.

```python
# Save the trained pyTorch model weight
torch.save(model.state_dict(), 'model/sine_mlp3.pth')
```

### Transform pyTorch Model for TVM

>**File:** `0103scriptedpyTorch.py`
**Info:** [Transform pyTorch model](https://tvm.apache.org/docs/how_to/compile_models/from_pytorch.html).

```python
# Just-in-time compilation
input_shape = [1,1]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

scripted_model.save('model/sine_mlp3_scripted.pth')
```

## Step 2: Compiled with TVM to Generate Library

>**File:** `0201tvm.py`\
**Info:** .

```python
import tvm
from tvm import relay
from tvm.relay.backend import Executor
from tvm.relay.backend import Runtime
from tvm.driver import tvmc
from tvm.micro import export_model_library_format

RUNTIME = Runtime('crt', {'system-lib':False})
EXECUTOR = Executor('aot',
    {"unpacked-api": True, 
    "interface-api": "c", 
    "workspace-byte-alignment": 4,
    "link-params": True,},
    )

TARGET = tvm.target.target.stm32('stm32F7xx')

# Build Model in TVM Relay
with tvm.transform.PassContext(opt_level=3, 
    config={"tir.disable_vectorize": True, 
    "tir.usmp.enable": True}):
    module = relay.build(model.mod, 
                        target=TARGET, 
                        runtime=RUNTIME, 
                        params=model.params, 
                        executor=EXECUTOR)

export_model_library_format(module, './models/default/default.tar')

```

## Step 3: Compile with RIOT OS

### 3.1 Create Code for Embedding Model in C

>**File:** `0301sin.c`\
**Info:** .

* Write `0301sin.c`

```c
#include <tvmgen_default.h>
#include "model_io_vars.h"

int ch=0;
scanf("%d", &ch);
input = (float)(ch*1.0/100);
tvmgen_default_run(&default_inputs, &default_outputs);
printf("Input: %d, output: %d\n", (int)(input*100), (int)(output*100));
```

* Write `model_io_vars.h`
```c
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
```

* `Input and output` Refer to Model Code generated by TVM. \
Source : `./models/default/default.tar/codegen/host/include/tvmgen_default.h`
```c
/*!
 * \brief Input tensor pointers for TVM module "default" 
 */
struct tvmgen_default_inputs {
  void* input0;
};

/*!
 * \brief Output tensor pointers for TVM module "default" 
 */
struct tvmgen_default_outputs {
  void* output;
};

/*!
 * \brief entrypoint function for TVM module "default"
 * \param inputs Input tensors for the module 
 * \param outputs Output tensors for the module 
 */
int32_t tvmgen_default_run(
  struct tvmgen_default_inputs* inputs,
  struct tvmgen_default_outputs* outputs
);
```

### 3.2 Download RIOT OS and Write `Makefile`

[RIOT OS Web](https://doc.riot-os.org/) for information.\
[RIOT OS CODE](https://github.com/RIOT-OS/RIOT) for testing.


```shell
git clone https://github.com/RIOT-OS/RIOT.git 
```

* Create `Makefile` in `ROOT` Directory

```Makefile
RIOTBASE= ./RIOT

BOARD ?= stm32f746g-disco
APPLICATION = SINE

EXTERNAL_PKG_DIRS += models

USEPKG += default 
USEMODULE += stdin

include $(RIOTBASE)/Makefile.include

CFLAGS += -Wno-strict-prototypes 
CFLAGS += -Wno-missing-include-dirs

override BINARY := $(ELFFILE)

```

* Create `Makefile` in `./models/default` Directory

```Makefile
include $(RIOTBASE)/makefiles/utvm.inc.mk
```

* Create `utvm.inc.mk` in `./RIOT/makefiles` Directory

```Makefile
_CURDIR := $(shell basename $(CURDIR))
UTVM_NAME ?= $(_CURDIR)
UTVM_DIR_BASE ?= $(BINDIR)/utvm
UTVM_MODEL_DIR = $(UTVM_DIR_BASE)/$(UTVM_NAME)

all:
	$(QQ)"$(MAKE)" -C $(UTVM_MODEL_DIR)/codegen/host/src -f $(RIOTBASE)/makefiles/utvm/Makefile.utvm \
		UTVM_MODULE_NAME=$(UTVM_NAME) UTVM_NAME=$(UTVM_NAME) UTVM_MODEL_DIR=$(UTVM_MODEL_DIR)

prepare: $(CURDIR)/$(UTVM_NAME).tar
	mkdir -p $(UTVM_MODEL_DIR)
	tar --extract --file=$< --directory $(UTVM_MODEL_DIR) --touch

clean:
	$(QQ)rm -Rf $(UTVM_MODEL_DIR)
```

* Create `Makefile.utvm` in `./RIOT/makefiles/utvm` Directory


```Makefile
MODULE := $(UTVM_MODULE_NAME)

CFLAGS += -I$(UTVM_MODEL_DIR)/runtime/include
CFLAGS += -I$(UTVM_MODEL_DIR)/codegen/host/include
CFLAGS += -Wno-pedantic 
CFLAGS += -Wno-attributes
CFLAGS += -Wno-incompatible-pointer-types
CFLAGS += -Wno-cast-align
CFLAGS += -Wno-unused-parameter
CFLAGS += -Wno-unused-variable

include  $(RIOTBASE)/Makefile.base
```

## Step 4: Compile, Flash and Test

```shell
make flash

make term
```

> Thanks to [U-TOE](https://github.com/zhaolanhuang/U-TOE) project, and [RIOT](https://github.com/RIOT-OS/RIOT) project.



