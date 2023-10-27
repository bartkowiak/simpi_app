The code you provided appears to be a simulation of a ticket selling process in a public transportation system. It simulates passenger arrivals, queuing, and ticket purchasing at both ticket vending machines (Fahrscheinautomaten) and Touch & Go machines (T&G). The simulation is run multiple times to collect statistical data. Here is an explanation of the key functions and how they work:

1. `ticket_verkauf_sim`: This is the main simulation function. It takes multiple parameters that define the simulation, such as the number of ticket vending machines, arrival rates, service rates, simulation time, and peak hours. This function uses the SimPy library to simulate passenger behavior. The simulation results are stored in various dictionaries and lists.

2. `generate_fg`: This function generates random passenger arrival times based on an exponential distribution. The rate of arrival is controlled by the parameter `FGZ_PRO_MIN`.

3. `assign_fsa`: It generates random service times at Fahrscheinautomaten (ticket vending machines) based on an exponential distribution. The rate of service is controlled by the parameter `TICKET_PRO_MIN`.

4. `assign_tag`: This function models passenger choices between using a Fahrscheinautomat (FSA) or a Touch-and-Go (T&G) machine. The choice is determined based on the probability `P`.

5. `fg`: This function models the behavior of a single passenger. It calculates queue lengths for each type of machine, makes a choice, and records various statistics like waiting times and service times.

6. `fsa_run`: This function simulates passengers arriving at the station, generating them at varying rates during peak hours and normal hours. It sends each passenger to the `fg` function for processing.

7. `NoInSystem`: A utility function that calculates the total number of customers in the resource (queue).

8. `mc_sim`: This function performs a Monte Carlo simulation by running multiple simulations (`NUM_RUNS` times) and collecting the results. It returns the results as dictionaries.

9. `create_output`: This function organizes the simulation results into a structured DataFrame, including passenger arrival times, service start times, chosen machines, service times, waiting times, and information about finished passengers. It also calculates various statistics and metrics.

10. `bericht`: This function generates a report based on simulation results, including information about passenger behavior, queue length, and average waiting times.

11. `plots`: This function generates plots to visualize simulation results, such as histograms and bar charts of waiting times.

12. `video`: This function creates a video or static visualization of queue evolution during the simulation.

13. `run_sim`: This is the main function to run the simulation and generate reports and plots. It takes additional parameters, such as the number of runs and the video flag.

Overall, this code is a comprehensive simulation of a ticket selling process in a public transportation system and provides the means to analyze and visualize the results of the simulation.


ticket_verkauf_sim
   ├── generate_fg
   ├── assign_fsa
   └── assign_tag

create_output
   ├── df_converter

mc_sim
   ├── ticket_verkauf_sim
   ├── create_output

bericht

plots
   ├── create_output

video
   ├── create_output

run_sim
   ├── mc_sim
       ├── ticket_verkauf_sim
       └── create_output
   ├── bericht
   ├── plots
   └── video
       ├── create_output
