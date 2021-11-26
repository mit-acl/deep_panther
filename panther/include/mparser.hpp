// Partly taken from section 1.4 of https://caiorss.github.io/C-Cpp-Notes/embedded_scripting_languages.html

#ifndef _MPARSER_HPP_
#define _MPARSER_HPP_

#include <string>
#include <memory>
#include <functional>

// Note: PimPl (Pointer-To-Implementation) idiom
// is a technique for reducing compile-time

class MathEvaluator
{
  struct impl;
  std::shared_ptr<impl> m_pimpl;

public:
  MathEvaluator();
  ~MathEvaluator();  // = default;
  MathEvaluator& add_var(std::string name, double& ref);

  // Register function pointer callback or non-capture lambda
  MathEvaluator& add_function(std::string, double fptr(double));
  // Register function of two variables
  MathEvaluator& add_function(std::string, double fptr(double, double));

  double eval_code(std::string code);
  bool compile(std::string code);
  double value() const;
  void repl();
};

#endif