def temperature_linear(args, total_step):
    temperature = args.temperature_begin
    while True:
        temperature += (args.temperature_end-args.temperature_begin)/total_step
        yield temperature

def temperature_constant(args, total_step):
    temperature = args.temperature_constant
    while True:
        yield temperature



