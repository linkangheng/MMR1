def temperature_linear(begin, end, total_step):
    temperature = begin
    while True:
        temperature += (end-begin)/total_step
        yield temperature

def temperature_const(const):
    yield const



