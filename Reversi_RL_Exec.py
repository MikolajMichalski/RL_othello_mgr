from RL_othello_mgr import Test, testMinMax, Testhuman

valid_input = False
while valid_input == False:

    number_of_layers_input = input("Wybierz architekturę sieci neuronowej: \n"
                        "1) 2 warstwy ukryte\n"
                        "2) 3 warstwy ukryte\n"
                        "3) 4 warstwy ukryte\n")

    try:
        num_layers_int = int(number_of_layers_input)
        if num_layers_int < 1 or num_layers_int > 3:
            print("Wprowadzono błędne dane!!!\n")
            continue
        valid_input = True
    except ValueError:
        print("Wprowadzono błędne dane!!!\n")
        continue

numberOfLayers = num_layers_int +1

valid_input1 = False
while valid_input1 == False:

    opponent_agent_type = input("Wybierz rodzaj przeciwnika: \n"
                        "1) przeciwnik losowy\n"
                        "2) przeciwnik MinMax\n"
                        "3) człowiek\n")

    try:
        opponent_int_input = int(opponent_agent_type)
        if opponent_int_input < 1 or opponent_int_input > 3:
            print("Wprowadzono błędne dane!!!\n")
            continue
        valid_input1 = True
        if opponent_int_input == 2:
            valid_input3 = False

            while valid_input3 == False:

                min_max_depth_input = input("Wybierz głębokość MinMax: \n")

                try:
                    min_max_depth_int = int(min_max_depth_input)
                    if min_max_depth_int < 1:
                        print("Wprowadzono błędne dane!!!\n")
                        continue
                    valid_input3 = True
                except ValueError:
                    print("Wprowadzono błędne dane!!!\n")
                    continue
            min_max_depth = min_max_depth_int

    except ValueError:
        print("Wprowadzono błędne dane!!!\n")
        continue

valid_input4 = False
while valid_input4 == False:

    verbose_input = input("Włączyć tryb podglądu?: \n"
                          "1) TAK\n"
                          "2) NIE\n")

    try:
        verbose_input_int = int(verbose_input)
        if verbose_input_int < 1 or verbose_input_int > 2:
            print("Wprowadzono błędne dane!!!\n")
            continue
        valid_input4 = True
    except ValueError:
        print("Wprowadzono błędne dane!!!\n")
        continue


if verbose_input_int == 1:
    verbose = True
else:
    verbose = False

if opponent_int_input == 1:
    Test.start(numberOfLayers, verbose)
elif opponent_int_input == 2:
    testMinMax.start(numberOfLayers, min_max_depth_int, verbose)
elif opponent_int_input == 3:
    Testhuman.start(numberOfLayers, verbose)

