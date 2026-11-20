#include "repl.hpp"
#include "session.hpp"
#include <iostream>

int main(){
    hopfield::Session session;
    hopfield::runRepl(session, std::cin, std::cout);
}