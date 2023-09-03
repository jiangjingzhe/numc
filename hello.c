#include<Python.h>
#include "CUnit/CUnitCI.h"

/* Test that one equals one */
static void test_simple_pass1(void) {
    CU_ASSERT_FATAL(1 == 1);
}

CUNIT_CI_RUN("my-suite",
             CUNIT_CI_TEST(test_simple_pass1));
