bin_PROGRAMS = CreateOnDiskPt 
CreateOnDiskPt_SOURCES = Main.cpp
AM_CPPFLAGS = -W -Wall -ffor-scope -D_FILE_OFFSET_BITS=64 -D_LARGE_FILES -DUSE_HYPO_POOL -I$(top_srcdir)/moses/src $(BOOST_CPPFLAGS)

CreateOnDiskPt_LDADD = -L$(top_srcdir)/OnDiskPt/src -L$(top_srcdir)/moses/src -lOnDiskPt -lmoses $(BOOST_THREAD_LDFLAGS) $(BOOST_THREAD_LIBS)  @KENLM_LDFLAGS@
CreateOnDiskPt_DEPENDENCIES = $(top_srcdir)/OnDiskPt/src/libOnDiskPt.a $(top_srcdir)/moses/src/libmoses.la @KENLM_DEPS@



