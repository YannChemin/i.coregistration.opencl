MODULE_TOPDIR = ../..

PGM = i.coregistration.opencl

LIBES = $(IMAGERYLIB) $(VECTORLIB) $(DBMILIB) $(RASTERLIB) $(GISLIB) \
	$(MATHLIB)
DEPENDENCIES = $(IMAGERYDEP) $(VECTORDEP) $(DBMIDEP) $(RASTERDEP) \
	$(GISDEP)
EXTRA_INC = $(VECT_INC) $(OCLINCPATH) $(OPENMP_INCPATH) -I$(OBJDIR)
EXTRA_CFLAGS = $(VECT_CFLAGS) $(OPENMP_CFLAGS) -O3 -std=gnu11
EXTRA_LIBS = $(OCLLIBPATH) $(OCLLIB) $(OPENMP_LIBPATH) $(OPENMP_LIB) -lcurl

include $(MODULE_TOPDIR)/include/Make/Module.make

default: cmd

# Embed the OpenCL kernels as C string literals.
$(OBJDIR)/ocl_backend.o: $(OBJDIR)/ops_cl.h $(OBJDIR)/disk_kernels_cl.h

$(OBJDIR)/%_cl.h: %.cl | $(OBJDIR)
	sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/' $< > $@

# Developer check of the generic kernels against PyTorch (not installed):
# make ops_check, then tests/ops_compare.py on its output.
ops_check: $(OBJDIR)/ops_check

$(OBJDIR)/ops_check: tests/ops_check.c $(OBJDIR)/ops.o $(OBJDIR)/ocl_backend.o
	$(CC) $(ALL_CFLAGS) -I. -o $@ $< $(OBJDIR)/ops.o $(OBJDIR)/ocl_backend.o \
		$(LDFLAGS) $(LIBES) $(EXTRA_LIBS) $(MATHLIB)

.PHONY: ops_check

# Developer check of the robust estimation stage (CPU only, not installed).
robust_check: $(OBJDIR)/robust_check

$(OBJDIR)/robust_check: tests/robust_check.c $(OBJDIR)/robust.o \
		$(OBJDIR)/models.o $(OBJDIR)/maxflow.o
	$(CC) $(ALL_CFLAGS) -I. -o $@ $< $(OBJDIR)/robust.o $(OBJDIR)/models.o \
		$(OBJDIR)/maxflow.o $(LDFLAGS) $(LIBES) $(EXTRA_LIBS) $(MATHLIB)

.PHONY: robust_check
