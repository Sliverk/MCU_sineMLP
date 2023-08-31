RIOTBASE= ./RIOT

BOARD ?= stm32f746g-disco
APPLICATION = SINE

EXTERNAL_PKG_DIRS += pkg

USEPKG += sineMLP 
USEMODULE += stdin

include $(RIOTBASE)/Makefile.include

CFLAGS += -Wno-strict-prototypes 
CFLAGS += -Wno-missing-include-dirs

override BINARY := $(ELFFILE)

# list-ttys-json:
# 	$(Q) python $(RIOTTOOLS)/usb-serial/ttys.py --format json
