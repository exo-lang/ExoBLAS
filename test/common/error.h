#pragma once

class UnsupportedParameterException : public std::runtime_error {
 public:
  UnsupportedParameterException(char const* const message) throw()
      : std::runtime_error(message) {}
  virtual char const* what() const throw() { return exception::what(); }
};
