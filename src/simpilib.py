
import numpy as np
import pandas as pd
import simpy 
import matplotlib.pyplot as plt
import itertools
from IPython import display
from ipywidgets import interactive
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import warnings
import seaborn as sns

def ticket_verkauf_sim(NUM_FSA,NUM_TG,P,FGZ_PRO_MIN_NORMAL,FGZ_PRO_MIN_SZ,TICKET_PRO_MIN,TICKET_PRO_MIN_TG,SIM_TIME,STOSSZEITEN):
    """
    Simulate a ticket selling process in a public transportation system with various parameters.

    Parameters:
    NUM_FSA (int): Number of Fahrscheinautomaten (ticket vending machines).
    NUM_TG (int): Number of Touch & Go (T&G) machines.
    P (float): Probability of choosing T&G machines.
    FGZ_PRO_MIN_NORMAL (float): Arrival rate of passengers after peak hours (per minute).
    FGZ_PRO_MIN_SZ (float): Arrival rate of passengers during peak hours (per minute).
    TICKET_PRO_MIN (float): Ticket selling rate at Fahrscheinautomaten (per minute).
    TICKET_PRO_MIN_TG (float): Ticket selling rate at T&G machines (per minute).
    SIM_TIME (float): Total simulation time (in minutes).
    STOSSZEITEN (float): Duration of peak hours (in minutes).

    Returns:
    wait_t (dict): Dictionary with passenger waiting times.
    queues (list): List of queue sizes at different time points.
    arr_t (list): List of arrival times of passengers.
    fg_ankommen (dict): Dictionary of passenger arrival times.
    service_t (dict): Dictionary of service times.
    start_t (dict): Dictionary of start times for service.
    choices (dict): Dictionary of machine choices made by passengers.
    fg_beended (dict): Dictionary of finished passengers.

    This function simulates a ticket selling process in a public transportation system. It models passenger arrivals, queuing, and ticket purchasing at Fahrscheinautomaten and Touch & Go machines. The simulation is run for a specified duration (SIM_TIME) with given parameters. Various statistics and data are collected and returned as dictionaries and lists.

    Args:
    - NUM_FSA and NUM_TG represent the number of ticket vending machines and Touch & Go machines available.
    - P is the probability of passengers choosing T&G machines.
    - FGZ_PRO_MIN_NORMAL and FGZ_PRO_MIN_SZ are the arrival rates of passengers during normal and peak hours, respectively.
    - TICKET_PRO_MIN and TICKET_PRO_MIN_TG are the ticket selling rates at Fahrscheinautomaten and T&G machines.
    - SIM_TIME is the total simulation time, and STOSSZEITEN is the duration of peak hours.

    The function uses a discrete-event simulation approach to model passenger behaviors and machine interactions. The simulation results are collected and returned as dictionaries and lists.

    """

    import random

    def generate_fg(FGZ_PRO_MIN):
        """
        Generate passenger arrival times based on an exponential distribution.
        Parameters:
        FGZ_PRO_MIN (float): Arrival rate of passengers (per minute).
        Returns:
        float: A randomly generated passenger arrival time.
        This function models passenger arrivals in a public transportation system. It uses an exponential distribution to generate random arrival times for passengers. The rate parameter FGZ_PRO_MIN defines how frequently passengers arrive.
        Args:
        - FGZ_PRO_MIN: The arrival rate of passengers per minute, which influences the frequency of arrivals.
        Returns:
        - A randomly generated passenger arrival time (in minutes) based on the exponential distribution.
        Example usage:
        generated_time = generate_fg(10.0)  # Generates a random arrival time with an arrival rate of 10 passengers per minute.
        """
        import numpy as np

        # Generate a random passenger arrival time based on the exponential distribution
        arrival_time = np.random.exponential(1.0 / FGZ_PRO_MIN)
        return arrival_time


    def assign_fsa(TICKET_PRO_MIN):
        """
        Generate service times at Fahrscheinautomaten (ticket vending machines) based on an exponential distribution.
        Parameters:
        TICKET_PRO_MIN (float): Ticket selling rate at Fahrscheinautomaten (per minute).
        Returns:
        float: A randomly generated service time.
        This function models the time it takes to serve a passenger at a Fahrscheinautomat. It uses an exponential distribution to generate random service times. The rate parameter TICKET_PRO_MIN defines how quickly tickets are sold at the machine.
        Args:
        - TICKET_PRO_MIN: The ticket selling rate at Fahrscheinautomaten (tickets per minute), which determines the speed of service.
        Returns:
        - A randomly generated service time (in minutes) based on the exponential distribution.
        Example usage:
        generated_time = assign_fsa(2.5)  # Generates a random service time at a Fahrscheinautomat with a rate of 2.5 customers served per minute.
        """
        # Generate a random service time based on the exponential distribution
        service_time = np.random.exponential(1.0 / TICKET_PRO_MIN)
        return service_time

    def assign_tag(P):
        """
        Choose a Touch-and-Go (T&G) machine with a given probability.
        Parameters:
        P (float): Probability of choosing a T&G machine.
        Returns:
        int: 0 for choosing a Fahrscheinautomat (FSA), 1 for choosing a T&G machine.
        This function models the selection of a ticket vending machine, either a Fahrscheinautomat (FSA) or a Touch-and-Go (T&G) machine. The probability parameter P determines the likelihood of choosing a T&G machine.
        Args:
        - P: Probability of choosing a T&G machine. Should be a value between 0 and 1.
        Returns:
        - 0 for choosing an FSA (Fahrscheinautomat).
        - 1 for choosing a T&G machine (Touch-and-Go).

        Example usage:
        choice = assign_tag(0.3)  # Chooses a ticket vending machine with a 30% probability of selecting a T&G machine.
        """

        # Choose a vending machine based on the provided probability
        choice = np.random.choice(2, 1, p=[1 - P, P])[0]
        return choice




    wait_t = {}  # Dictionary to store passenger waiting times
    queues = []  # List to store queue sizes over time
    arr_t = []  # List to store passenger arrival times
    fg_ankommen = {}  # Dictionary to store passenger arrival times
    service_t = {}  # Dictionary to store service times
    start_t = {}  # Dictionary to store service start times
    choices = {}  # Dictionary to store machine choices made by passengers
    fg_beended = {}  # Dictionary to store finished passengers

    def fg(env,fg, resourcen):
            """
            Simulate the behavior of a passenger in the public transportation system.
            Parameters:
            env (simpy.Environment): Simpy environment for event simulation.
            fg (int): Passenger identifier.
            resourcen (dict): Dictionary of resources representing ticket vending machines.
            This function models the behavior of a passenger from their arrival at the station to their service and departure. It takes into account the choice of ticket vending machine and service times.
            Args:
            - env: Simpy environment used for event scheduling and time tracking.
            - fg: Passenger identifier.
            - resourcen: Dictionary of resources representing ticket vending machines.
            Returns:
            None
            During the simulation, this function records information about the passenger's arrival, choice of machine, wait time, and service time. It also updates queue lengths for each machine and service times.
            Example usage:
            env.process(fg(env, 1, resourcen))  # Simulate the behavior of a passenger with identifier 1.
            """
            # Save arrival of FG
            t_arrival = env.now
            fg_ankommen[fg] = t_arrival

            # Calculate queue lengths for Fahrscheinautomaten (FSA) and Touch-and-Go (T&G) machines
            QlengthFSA = {i:NoInSystem(resourcen[i]) for i in range(NUM_FSA)}
            QlengthTG = {i:NoInSystem(resourcen[i]) for i in range(NUM_FSA,NUM_FSA+NUM_TG)}

            # Record the queue lengths for all machines

            queues.append({q:len(resourcen[q].put_queue) for q in range(len(resourcen))})
            
            # Determine whether the passenger chooses T&G or FSA based on probability
            
            touch_and_go=assign_tag(P)
            
            # Choose a machine based on queue lengths and choice
            if NUM_TG==0 or touch_and_go==0:
                touch_and_go=assign_tag(0)
                choice=[k for k,v in sorted(QlengthFSA.items(), key=lambda a:a[1])][0]
            else:

                choice=[k for k,v in sorted(QlengthTG.items(), key=lambda a:a[1])][0]


            # Record the passenger's choice
            choices[fg] = choice
            
            # Request and occupy the chosen machine

            with resourcen[choice].request() as request:              
                yield request
                # Calculate and record the start and waiting time

                t_service=env.now
                start_t[fg]=(t_service)

                wait_t[fg]=(t_service-t_arrival)
                # Determine service time based on machine type
                if NUM_TG==0 or touch_and_go==0:
                    yield env.timeout(assign_fsa(TICKET_PRO_MIN))
                else:
                    yield env.timeout(assign_fsa(TICKET_PRO_MIN_TG))
                # Record updated queue lengths

                queues.append({i: len(resourcen[i].put_queue) for i in range(len(resourcen))})

                t_depart=env.now
                service_t[fg]=(t_depart-t_service)

                fg_beended[fg]=fg


    def fsa_run(env):
        """
        Simulate passengers arriving at the station and joining ticket vending machine queues.
        Parameters:
        env (simpy.Environment): Simpy environment for event simulation.
        This function models the arrival of passengers at the station. It generates passengers at a rate that varies depending on whether it's during peak hours (STOSSZEITEN) or normal hours.
        Args:
        - env: Simpy environment used for event scheduling and time tracking.
        Returns:
        None
        During the simulation, passengers are generated at a rate specified by `generate_fg` function. The rate depends on whether it's during peak hours (FGZ_PRO_MIN_SZ) or normal hours (FGZ_PRO_MIN_NORMAL). Passengers are then sent to the `fg` function for further processing.
        Example usage:
        env.process(fsa_run(env))  # Start simulating passenger arrivals at the station.
        """
        i=0
        while True:
            i+=1
            # Determine the arrival rate based on peak hours or normal hours
            if env.now<=STOSSZEITEN:
                yield env.timeout(generate_fg(FGZ_PRO_MIN_SZ)) # Generate passengers during peak hours
            else:
                yield env.timeout(generate_fg(FGZ_PRO_MIN_NORMAL)) # Generate passengers during normal hours
            # Send each generated passenger to the 'fg' function for further processing
            env.process(fg(env,i, resourcen))


    def NoInSystem(f):
        """Total number of customers in the resource R"""
        return max([0, len(f.put_queue) + len(f.users)])



    ###Simulation part

    #np.random.seed(RANDOM_SEED) 
    np.random.seed() 

    env=simpy.Environment()
    #generate resources
    resourcen={i: simpy.Resource(env) for i in range(NUM_FSA+NUM_TG)}

    #run
    env.process(fsa_run(env))
    env.run(until=SIM_TIME)


    return wait_t, queues , arr_t, fg_ankommen,service_t,start_t, choices, fg_beended    

def create_output(fg_ankommen, start_t, choices, service_t, wait_t, fg_beended, SIM_TIME, NUM_FSA, NUM_TG):
    """
    Create a summary DataFrame of the simulation results.

    Parameters:
    fg_ankommen (dict): Dictionary of passenger arrival times.
    start_t (dict): Dictionary of service start times.
    choices (dict): Dictionary of the chosen ticket vending machines.
    service_t (dict): Dictionary of service times.
    wait_t (dict): Dictionary of passenger waiting times.
    fg_beended (dict): Dictionary of finished passengers.
    SIM_TIME (float): Total simulation time.
    NUM_FSA (int): Number of ticket vending machines (Fahrscheinautomaten).
    NUM_TG (int): Number of Touch and Go machines.

    Returns:
    pd.DataFrame: A DataFrame containing the summary of simulation results.

    This function organizes the simulation results into a structured DataFrame, including passenger arrival times, service start times, chosen machines, service times, waiting times, and information about finished passengers. It also calculates statistics and metrics related to the simulation.

    Args:
    - fg_ankommen: Dictionary of passenger arrival times.
    - start_t: Dictionary of service start times.
    - choices: Dictionary of the chosen machines (FSA or T&G).
    - service_t: Dictionary of service times.
    - wait_t: Dictionary of waiting times.
    - fg_beended: Dictionary of finished passengers.
    - SIM_TIME: Total simulation time.
    - NUM_FSA: Number of ticket vending machines (Fahrscheinautomaten).
    - NUM_TG: Number of Touch and Go machines.

    Returns:
    - A summary DataFrame containing simulation results and calculated metrics.

    Example usage:
    summary_df = create_output(fg_ankommen, start_t, choices, service_t, wait_t, fg_beended, SIM_TIME, NUM_FSA, NUM_TG)
    """

    # Define a helper function to convert a dictionary to a DataFrame
    def df_converter(data_dict, column_name):
        temp = pd.DataFrame(data_dict, index=[0]).T.reset_index()
        temp.columns = ['fg', column_name]
        return temp

    list_columns = ['arr_t', 'start_t', 'fsa', 'service_t', 'wait_t', 'fg_beended']
    list_dicts = [fg_ankommen, start_t, choices, service_t, wait_t, fg_beended]

    # Create a dictionary to store DataFrames for each result category
    result_dict = {}
    for i, data in enumerate(list_dicts):
        result_dict[i] = df_converter(data, list_columns[i])

    # Merge the DataFrames based on the 'fg' column
    df = result_dict[0]
    for i in range(1, len(result_dict)):
        df = df.merge(result_dict[i], how='left', on='fg')

    # Calculate additional metrics and statistics
    fg_ohne_t = len(df[df['wait_t'].isna()])
    ratio_fg_sch = len(df[df['wait_t'].isna()]) / len(df) * 100 if fg_ohne_t > 0 else 0

    df['ratio_fg_sch'] = ratio_fg_sch
    df['fg_ohne_t'] = fg_ohne_t

    df['wait_t_cl'] = np.where(df['wait_t'].isna(), SIM_TIME - df['arr_t'], df['wait_t'])
    df['start_t_cl'] = np.where(((df['service_t'].isna()) & (~df['service_t'].isna())), SIM_TIME - df['start_t'], df['start_t'])

    # Create columns for each ticket vending machine (FSA or T&G)
    for i in range(NUM_FSA + NUM_TG):
        df['start_t_' + str(i)] = np.where(df['fsa'] == i, df['start_t_cl'], 0)
        df['service_t_' + str(i)] = np.where(df['fsa'] == i, df['service_t'], 0)

    df['wait_mean_sr'] = df['wait_t'].mean()

    return df

def mc_sim(NUM_FSA, NUM_TG, P, FGZ_PRO_MIN_NORMAL, FGZ_PRO_MIN_SZ, TICKET_PRO_MIN, TICKET_PRO_MIN_TG, SIM_TIME, STOSSZEITEN, NUM_RUNS=100):

    """
    Perform Monte Carlo simulation by running multiple simulations and collecting the results.

    Parameters:
    NUM_FSA (int): Number of ticket vending machines (Fahrscheinautomaten).
    NUM_TG (int): Number of Touch and Go machines.
    P (float): Probability of choosing a Touch and Go machine.
    FGZ_PRO_MIN_NORMAL (float): Arrival rate of passengers after peak hours.
    FGZ_PRO_MIN_SZ (float): Arrival rate of passengers during peak hours.
    TICKET_PRO_MIN (float): Ticket selling rate at Fahrscheinautomaten (per minute).
    TICKET_PRO_MIN_TG (float): Ticket selling rate at Touch and Go machines (per minute).
    SIM_TIME (float): Total simulation time.
    STOSSZEITEN (float): Duration of peak hours.
    NUM_RUNS (int, optional): Number of simulation runs. Defaults to 100.

    Returns:
    Tuple[dict, dict]: A tuple containing dictionaries of simulation results and queue information.

    This function conducts Monte Carlo simulation by running multiple simulations and collecting the results. It simulates passenger behavior at ticket vending machines under different scenarios, such as peak hours and machine choices.

    Args:
    - NUM_FSA: Number of ticket vending machines (Fahrscheinautomaten).
    - NUM_TG: Number of Touch and Go machines.
    - P: Probability of choosing a Touch and Go machine.
    - FGZ_PRO_MIN_NORMAL: Arrival rate of passengers after peak hours.
    - FGZ_PRO_MIN_SZ: Arrival rate of passengers during peak hours.
    - TICKET_PRO_MIN: Ticket selling rate at Fahrscheinautomaten (per minute).
    - TICKET_PRO_MIN_TG: Ticket selling rate at Touch and Go machines (per minute).
    - SIM_TIME: Total simulation time.
    - STOSSZEITEN: Duration of peak hours.
    - NUM_RUNS: Number of simulation runs (default is 100).

    Returns:
    - A tuple containing two dictionaries: one for simulation results and one for queue information.

    Example usage:
    simulation_results, queues_info = mc_sim(NUM_FSA, NUM_TG, P, FGZ_PRO_MIN_NORMAL, FGZ_PRO_MIN_SZ, TICKET_PRO_MIN, TICKET_PRO_MIN_TG, SIM_TIME, STOSSZEITEN)
    """
    output_sim = {}  # Dictionary to store simulation results
    queues_dict = {}  # Dictionary to store queue information

    for i in range(NUM_RUNS):
        # Run a single simulation
        wait_t, queues, arr_t, fg_ankommen, service_t, start_t, choices, fg_beended = ticket_verkauf_sim(NUM_FSA, NUM_TG, P, FGZ_PRO_MIN_NORMAL, FGZ_PRO_MIN_SZ, TICKET_PRO_MIN, TICKET_PRO_MIN_TG, SIM_TIME, STOSSZEITEN)

        # Create a summary of the simulation results
        output_sim[i] = create_output(fg_ankommen, start_t, choices, service_t, wait_t, fg_beended, SIM_TIME, NUM_FSA, NUM_TG)

        # Store queue information
        queues_dict[i] = queues

    return output_sim, queues_dict


def bericht(df_output, NUM_FSA, NUM_TG, P, FGZ_PRO_MIN_NORMAL, FGZ_PRO_MIN_SZ, TICKET_PRO_MIN, TICKET_PRO_MIN_TG, SIM_TIME, STOSSZEITEN):
    """
    Generate a report based on simulation results.

    Parameters:
    df_output (DataFrame): Simulation results as a DataFrame.
    NUM_FSA (int): Number of ticket vending machines (Fahrscheinautomaten).
    NUM_TG (int): Number of Touch and Go machines.
    P (float): Probability of choosing a Touch and Go machine.
    FGZ_PRO_MIN_NORMAL (float): Arrival rate of passengers after peak hours.
    FGZ_PRO_MIN_SZ (float): Arrival rate of passengers during peak hours.
    TICKET_PRO_MIN (float): Ticket selling rate at Fahrscheinautomaten (per minute).
    TICKET_PRO_MIN_TG (float): Ticket selling rate at Touch and Go machines (per minute).
    SIM_TIME (float): Total simulation time.
    STOSSZEITEN (float): Duration of peak hours.

    This function generates a report based on the simulation results. It includes information about passenger behavior, queue length, and average waiting times.

    Args:
    - df_output: Simulation results as a DataFrame.
    - NUM_FSA: Number of ticket vending machines (Fahrscheinautomaten).
    - NUM_TG: Number of Touch and Go machines.
    - P: Probability of choosing a Touch and Go machine.
    - FGZ_PRO_MIN_NORMAL: Arrival rate of passengers after peak hours.
    - FGZ_PRO_MIN_SZ: Arrival rate of passengers during peak hours.
    - TICKET_PRO_MIN: Ticket selling rate at Fahrscheinautomaten (per minute).
    - TICKET_PRO_MIN_TG: Ticket selling rate at Touch and Go machines (per minute).
    - SIM_TIME: Total simulation time.
    - STOSSZEITEN: Duration of peak hours.

    Example usage:
    bericht(simulation_results, NUM_FSA, NUM_TG, P, FGZ_PRO_MIN_NORMAL, FGZ_PRO_MIN_SZ, TICKET_PRO_MIN, TICKET_PRO_MIN_TG, SIM_TIME, STOSSZEITEN)
    """
    print('Anzahl FG pro Min in Stosszeiten: ' + str(FGZ_PRO_MIN_SZ))
    print('Anzahl FG pro Min nach Stosszeiten: ' + str(FGZ_PRO_MIN_NORMAL))

    print('Anteil der FG für T&G: ' + "{:.0%}".format(P))

    print('Anzahl FSA: ' + str(NUM_FSA))
    print('Anzahl T&G: ' + str(NUM_TG))

    print('Anzahl Tickets pro Minute bei FSA: ' + str(TICKET_PRO_MIN))
    print('Anzahl Tickets pro Minute bei T&G Gerät: ' + str(TICKET_PRO_MIN_TG))

    # Calculate and format average number of passengers without a ticket
    avg_passengers_without_ticket = np.ceil(df_output.fg_ohne_t.mean())
    percentage_passengers_without_ticket = format(df_output.ratio_fg_sch.mean(), ".2%")
    print(f'Anzahl FG ohne Ticket nach {SIM_TIME} Minuten mit {STOSSZEITEN} Minuten der Stosszeiten: {avg_passengers_without_ticket} ({percentage_passengers_without_ticket})')

    # Calculate and format average waiting time
    avg_waiting_time = format(df_output['wait_t_cl'].mean(), ".2f")
    print('Durchschnittliche Wartezeit in Minuten: ' + avg_waiting_time)

def plots(df_output, NUM_FSA, NUM_TG, NUM_RUNS):
    """
    Generate plots based on simulation results.

    Parameters:
    df_output (DataFrame): Simulation results as a DataFrame.
    NUM_FSA (int): Number of ticket vending machines (Fahrscheinautomaten).
    NUM_TG (int): Number of Touch and Go machines.
    NUM_RUNS (int): Number of simulation runs.

    This function generates various plots to visualize simulation results, including histograms and bar charts.

    Args:
    - df_output: Simulation results as a DataFrame.
    - NUM_FSA: Number of ticket vending machines (Fahrscheinautomaten).
    - NUM_TG: Number of Touch and Go machines.
    - NUM_RUNS: Number of simulation runs.

    Example usage:
    plots(simulation_results, NUM_FSA, NUM_TG, NUM_RUNS)
    """
    sns.set_style("whitegrid")

    color = ['orange', 'blue', 'green', 'yellow', 'red', 'brown', 'olive', 'purple', 'cyan', 'grey', 'pink']

    # Create subplots
    fig2, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    plt.grid()

    # Plot histogram of waiting times
    h = axes[0].hist(df_output.wait_t_cl, bins=20, rwidth=0.9, color='#b0dcf7')
    axes[0].axvline(np.mean(df_output.wait_t_cl), color='red', label='Mean')
    axes[0].text(np.mean(df_output.wait_t_cl) + 0.05, 0, 'Mean ' + format((np.mean(df_output.wait_t_cl)), ".2f"), rotation=90, color='red')
    axes[0].set_xlabel('Wartezeit[min]')
    axes[0].set_ylabel(f'Anzahl FG nach {NUM_RUNS} Simulationen')

    # Plot density plot of waiting times
    k = sns.distplot(df_output.wait_t_cl, ax=axes[1], color='#b0dcf7')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    axes[1].axvline(np.mean(df_output.wait_t_cl), color='red', label='Mean')
    axes[1].text(np.mean(df_output.wait_t_cl) + 0.05, 0, 'Mean ' + format((np.mean(df_output.wait_t_cl)), ".2f"), rotation=90, color='red')
    axes[1].set_xlabel('Wartezeit[min]')
    axes[1].set_ylabel('Wahrscheinlichkeitsdichte')

    # Create a bar chart to visualize queue time and service time
    fig = plt.figure(figsize=(25, 15))
    temp = df_output[df_output['sim_nr'] == 0]

    # Plot queue time and service time for each machine
    plt.barh(
        y=temp.fg,
        left=temp.arr_t,
        width=temp.wait_t_cl,
        alpha=1.0,
        color="gainsboro", label='Wartezeit')

    for i in range(NUM_FSA + NUM_TG):
        if i < NUM_FSA:
            label = 'Zeit beim FSA ' + str(i + 1)
        else:
            label = 'Zeit beim T&G ' + str(i + 1)
        plt.barh(
            y=temp.fg,
            left=temp['start_t_' + str(i)],
            width=temp['service_t_' + str(i)],
            alpha=1.0,
            color=color[i], label=label)

    plt.legend()
    plt.xlabel('Zeit[min]')
    plt.ylabel('FG Nummer')
    plt.show()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    plt.clf()

def video(queues_dict, df_output, NUM_FSA, NUM_TG, video_flag):
    """
    Create a video or static visualization of queue evolution.

    Parameters:
    queues_dict (dict): A dictionary containing queue information.
    df_output (DataFrame): Simulation results as a DataFrame.
    NUM_FSA (int): Number of ticket vending machines (Fahrscheinautomaten).
    NUM_TG (int): Number of Touch and Go machines.
    video_flag (str): 'Y' for video or any other value for a static plot.

    This function generates a video or static visualization to show how queues evolve during the simulation.

    Args:
    - queues_dict: A dictionary containing queue information.
    - df_output: Simulation results as a DataFrame.
    - NUM_FSA: Number of ticket vending machines (Fahrscheinautomaten).
    - NUM_TG: Number of Touch and Go machines.
    - video_flag: 'Y' for video or any other value for a static plot.

    Example usage:
    video(queues_dict, df_output, NUM_FSA, NUM_TG, 'Y')  # Create a video of queue evolution
    video(queues_dict, df_output, NUM_FSA, NUM_TG, 'N')  # Create a static plot of queue evolution
    """
    color = ['orange', 'blue', 'green', 'yellow', 'red', 'brown', 'olive', 'purple', 'cyan', 'grey', 'pink']

    q = pd.DataFrame(queues_dict[0])
    temp = df_output[df_output['sim_nr'] == 0]

    if video_flag == 'Y':

        x = []
        y = {}
        for j in range(NUM_FSA + NUM_TG):
            y[j] = []

        fig3, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        legend_list = []

        def animate(i):
            x.append(numbers[i])
            for j in range(NUM_FSA + NUM_TG):
                y[j].append((q[j][i]))
                axes.plot(x, y[j], color=color[j])
                if j < NUM_FSA:
                    legend_list.append('FSA' + str(j + 1))
                else:
                    legend_list.append('T&G' + str(j + 1))

            axes.legend(set(legend_list))

        plt.title("Warteschlangenaufbau")
        plt.xlabel('FG')
        plt.ylabel('Anzahl FG in der Schlange bei FSA/T&G')
        anim = animation.FuncAnimation(fig3, animate, frames=len(temp), interval=800, repeat=False)
        video = anim.to_html5_video()
        html = display.HTML(video)
        display.display(html)

        plt.close()
    else:
        legend_list = []
        pd.DataFrame(q).plot(figsize=(20, 5))

        for j in range(NUM_FSA + NUM_TG):
            if j < NUM_FSA:
                legend_list.append('FSA' + str(j + 1))
            else:
                legend_list.append('T&G' + str(j + 1))
        plt.legend(set(legend_list))

        plt.title("Warteschlangenaufbau")
        plt.xlabel('FG')
        plt.ylabel('Anzahl FG in der Schlange bei FSA/T&G')
        plt.show()



def run_sim(NUM_FSA,NUM_TG,P,FGZ_PRO_MIN_NORMAL,FGZ_PRO_MIN_SZ,TICKET_PRO_MIN,TICKET_PRO_MIN_TG,SIM_TIME,STOSSZEITEN,NUM_RUNS,video_flag='N'):
        """
        Run a simulation of a ticket selling process in a public transportation system with various parameters.

        Parameters:
        NUM_FSA (int): Number of Fahrscheinautomaten (ticket vending machines).
        NUM_TG (int): Number of Touch & Go (T&G) machines.
        P (float): Probability of choosing T&G machines.
        FGZ_PRO_MIN_NORMAL (float): Arrival rate of passengers after peak hours (per minute).
        FGZ_PRO_MIN_SZ (float): Arrival rate of passengers during peak hours (per minute).
        TICKET_PRO_MIN (float): Ticket selling rate at Fahrscheinautomaten (per minute).
        TICKET_PRO_MIN_TG (float): Ticket selling rate at T&G machines (per minute).
        SIM_TIME (float): Total simulation time (in minutes).
        STOSSZEITEN (float): Duration of peak hours (in minutes).
        NUM_RUNS (int): Number of simulation runs to perform.
        video_flag (str, optional): Flag to enable video presentation of queue data ('Y' for yes, 'N' for no).

        Returns:
        None

        This function performs a Monte Carlo simulation of a ticket selling process in a public transportation system. It simulates passenger arrivals, queuing, and ticket purchasing at Fahrscheinautomaten and Touch & Go machines. The simulation is run for multiple runs (specified by NUM_RUNS) to collect statistical data.

        After the simulation runs, it generates and displays various plots and statistics, including histograms of waiting times, average queue time, and a representation of the queue buildup over time.

        Args:
        - NUM_FSA and NUM_TG represent the number of ticket vending machines and Touch & Go machines available.
        - P is the probability of passengers choosing T&G machines.
        - FGZ_PRO_MIN_NORMAL and FGZ_PRO_MIN_SZ are the arrival rates of passengers during normal and peak hours, respectively.
        - TICKET_PRO_MIN and TICKET_PRO_MIN_TG are the ticket selling rates at Fahrscheinautomaten and T&G machines.
        - SIM_TIME is the total simulation time, and STOSSZEITEN is the duration of peak hours.
        - NUM_RUNS specifies the number of simulation runs to perform for statistical analysis.
        - video_flag can be set to 'Y' to enable a video presentation of queue data, or 'N' to disable it.
        """
        # The code for the simulation is defined within this function but is not included in this description.

        # Example usage:
        # RunSim(NUM_FSA=3, NUM_TG=2, P=0.3, FGZ_PRO_MIN_NORMAL=10, FGZ_PRO_MIN_SZ=20, TICKET_PRO_MIN=2, TICKET_PRO_MIN_TG=3, SIM_TIME=120, STOSSZEITEN=60, NUM_RUNS=10, video_flag='N')


        ###Simulation Pipeline ######                 

        #Running a Monte Carlo simulation with specified parameters and collecting simulation results.
        output_sim,queues_dict=mc_sim(NUM_FSA,NUM_TG,P,FGZ_PRO_MIN_NORMAL,FGZ_PRO_MIN_SZ,TICKET_PRO_MIN,TICKET_PRO_MIN_TG,SIM_TIME,STOSSZEITEN)

        #Preprocessing and aggregating the simulation results into a DataFrame.
        df_output=pd.concat(output_sim).reset_index().drop('level_1', axis=1).rename(columns={'level_0':'sim_nr'})

        #Printing a summary report based on the simulation results   
        bericht(df_output,NUM_FSA,NUM_TG,P,FGZ_PRO_MIN_NORMAL,FGZ_PRO_MIN_SZ,TICKET_PRO_MIN,TICKET_PRO_MIN_TG,SIM_TIME,STOSSZEITEN)

        #Creating and displaying various plots based on the simulation results
        plots(df_output,NUM_FSA,NUM_TG,NUM_RUNS)

        #Generating a video or static plot of queue evolution during the simulation.
        video(queues_dict,df_output,NUM_FSA,NUM_TG,video_flag='N')
