// Partly taken from section 1.4 of https://caiorss.github.io/C-Cpp-Notes/embedded_scripting_languages.html

#include "mparser.hpp"

#include <iostream>
#include <string>
#include <exprtk.hpp>

// --------------------------------------------------//

struct MathEvaluator::impl
{
  exprtk::expression<double> expr;
  exprtk::symbol_table<double> symbol_table;
  exprtk::parser<double> parser;
  std::string code;

  // Println function
  exprtk::rtl::io::println<double> println{};
};

// static method
// void MathEvaluator::dispose(impl* p)
// {
//   delete p;
// }

MathEvaluator::~MathEvaluator() = default;  // see https://www.fluentcpp.com/2017/09/22/make-pimpl-using-unique_ptr/

MathEvaluator::MathEvaluator() : m_pimpl(new impl)
{
  m_pimpl->symbol_table.add_constants();
  m_pimpl->expr.register_symbol_table(m_pimpl->symbol_table);
  m_pimpl->symbol_table.add_function("println", m_pimpl->println);
}

MathEvaluator& MathEvaluator::add_var(std::string name, double& ref)
{
  m_pimpl->symbol_table.add_variable(name, ref);
  return *this;
}

MathEvaluator& MathEvaluator::add_function(std::string name, double fptr(double))
{
  m_pimpl->symbol_table.add_function(name, fptr);
  return *this;
}

MathEvaluator& MathEvaluator::add_function(std::string name, double fptr(double, double))
{
  m_pimpl->symbol_table.add_function(name, fptr);
  return *this;
}

bool MathEvaluator::compile(std::string code)
{
  bool r = m_pimpl->parser.compile(code, m_pimpl->expr);

  if (!r)
  {
    std::string err = "Error: ";
    err = err + m_pimpl->parser.error();
    std::cerr << " Error: " << err << "\n";
    throw std::runtime_error(" [PARSER] Unable to parse expression.");
  }
  return r;
}

double MathEvaluator::value() const
{
  return m_pimpl->expr.value();
}

void MathEvaluator::repl()
{
  std::string line;
  double result;
  while (std::cin.good())
  {
    std::cout << " EXPRTK $ >> ";
    std::getline(std::cin, line);
    if (line.empty())
      continue;
    if (line == "exit")
      return;
    try
    {
      this->compile(line);
      std::cout << this->value() << '\n';
    }
    catch (std::runtime_error& ex)
    {
      std::cerr << "Error: " << ex.what() << '\n';
    }
  }
}