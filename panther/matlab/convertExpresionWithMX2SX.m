%Note that MX can only have numbers
function result=convertExpresionWithMX2SX(expression_MX, variables_MX, variables_SX)
    f = casadi.Function('f', {variables_MX} ,{expression_MX}, {'variables'} ,{'expression'});
    f = f.expand(); %It uses now SX
    result=f(variables_SX);
end