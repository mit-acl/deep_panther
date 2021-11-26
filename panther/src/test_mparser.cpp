// Taken from section 1.4 of https://caiorss.github.io/C-Cpp-Notes/embedded_scripting_languages.html

#include <iostream>
#include <string>
#include <cassert>
#include <vector>
#include "mparser.hpp"

double myfun(double a, double b);
void test_engine(MathEvaluator& engine, double& x);

int main(int argc, char** argv)
{
  std::puts(" [TRACE] I am up and running Ok. ");

  MathEvaluator engine;

  double x = 1.0, y = 2.0, z = 3.0;
  engine.add_var("x", x).add_var("y", y).add_var("z", z).add_function("myfun", &myfun);

  assert(argc == 2);

  auto command = std::string(argv[1]);
  if (command == "test")
  {
    test_engine(engine, x);
  }
  if (command == "repl")
  {
    engine.repl();
  }

  std::vector<MathEvaluator> s_mean;

  std::cerr << " [TRACE] Shutdown engine Ok. " << '\n';
  return 0;
}

// -----------------------------------------//

double myfun(double a, double b)
{
  std::cerr << "  [TRACE] a = " << a << "\n";
  std::cerr << "  [TRACE] b = " << b << "\n";
  double r = 3.0 * a + 5.0 * b;
  std::cerr << "  [TRACE] result = " << r << "\n";
  std::cerr << "---------------------------------\n";
  return r;
}

void test_engine(MathEvaluator& engine, double& x)
{
  std::string code = R"( 
        // Define local variables 
        var a := 2.0 / exp(x) * x^2 + y;
        var b := 10.0 * sqrt(x) + z;

        // println('\n => x = ', x);
        // println('\n => y = ', y);

        // Call custom function
        var k := myfun(x, y);

        // Comment: the last expression is returned 
        4.0 * a + 3 * b + 10 * z + k;        
    )";
  engine.compile(code);

  x = 3.0;
  std::cout << " => x = " << x << " ; engine = " << engine.value() << "\n";

  x = 5.0;
  std::cout << " => x = " << x << " ; engine = " << engine.value() << "\n";

  x = 15.0;
  std::cout << " => x = " << x << " ; engine = " << engine.value() << "\n";

  x = -15.0;
  std::cout << " => x = " << x << " ; engine = " << engine.value() << "\n";

  x = 20.0;
  std::cout << " => x = " << x << " ; engine = " << engine.value() << "\n";

  std::string code2 = R"( 
        // Vector/array variable 
        var xs [6] := {  2.0, 10.2,   -2.50,  9.256, 100.0,  25.0 };
        var ys [6] := { -2.0,  1.225, -5.56, 19.000, 125.0, 125.0 };

        println(' => xs =', xs);
        println(' => ys = ', ys);
        println(' => 3 * xs + 4 * ys = ', 3 * xs + 4 * ys);
        println(' => sum(xs) = ', sum(ys) );
        println(' => sum(ys) = ', sum(xs) );

    )";
  engine.compile(code2);
  engine.value();
}