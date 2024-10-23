use std::collections::VecDeque;
// use rand::Rng;
use rand_distr::{Distribution, Exp};
use plotters::prelude::*;

/// Structure to hold simulation results
#[derive(Default)]
struct SimulationResult {
    rho: f64,
    simulated_average: f64,
    theoretical_average: f64,
}
#[derive(Debug)]
enum EventType {
    Arrival,
    Departure,
}

#[derive(Debug)]
struct Event {
    time: f64,
    event_type: EventType,
}

struct Simulation {
    lambda: f64,     
    mu: f64,         
    rho: f64,        
    current_time: f64,
    customers_in_system: u32,
    event_queue: VecDeque<Event>,
    total_customers: u32,
    cumulative_time: f64,
    cumulative_customers: f64,
}

impl Simulation {
    /// Create a new simulation with given arrival and service rates
    fn new(lambda: f64, mu: f64) -> Self {
        let rho = lambda / mu;
        Simulation {
            lambda,
            mu,
            rho,
            current_time: 0.0,
            customers_in_system: 0,
            event_queue: VecDeque::new(),
            total_customers: 0,
            cumulative_time: 0.0,
            cumulative_customers: 0.0,
        }
    }

    /// Schedule an initial arrival event
    fn schedule_initial_arrival(&mut self) {
        let arrival_time = self.generate_interarrival_time();
        self.event_queue.push_back(Event {
            time: arrival_time,
            event_type: EventType::Arrival,
        });
    }

    /// Generate time until next arrival (exponential with rate lambda)
    fn generate_interarrival_time(&self) -> f64 {
        let exp = Exp::new(self.lambda).unwrap();
        exp.sample(&mut rand::thread_rng())
    }

    /// Generate service time (exponential with rate mu)
    fn generate_service_time(&self) -> f64 {
        let exp = Exp::new(self.mu).unwrap();
        exp.sample(&mut rand::thread_rng())
    }

    /// Update statistics when state changes
    fn update_stats(&mut self, time_delta: f64) {
        self.cumulative_time += time_delta;
        self.cumulative_customers += self.customers_in_system as f64 * time_delta;
    }

    /// Get average number of customers in system
    fn get_average_customers(&self) -> f64 {
        if self.cumulative_time > 0.0 {
            self.cumulative_customers / self.cumulative_time
        } else {
            0.0
        }
    }

    /// Get theoretical average number of customers
    fn get_theoretical_average(&self) -> f64 {
        self.rho / (1.0 - self.rho)
    }
    
    /// Handle an arrival event
    fn handle_arrival(&mut self, event_time: f64) {
        // Update statistics for time period since last event
        let time_delta = event_time - self.current_time;
        self.update_stats(time_delta);
        
        // Update current time and increment customers
        self.current_time = event_time;
        self.customers_in_system += 1;
        self.total_customers += 1;

        // Schedule next arrival
        let next_arrival_time = self.current_time + self.generate_interarrival_time();
        self.event_queue.push_back(Event {
            time: next_arrival_time,
            event_type: EventType::Arrival,
        });

        // If this is the only customer, schedule their departure
        if self.customers_in_system == 1 {
            let departure_time = self.current_time + self.generate_service_time();
            self.event_queue.push_back(Event {
                time: departure_time,
                event_type: EventType::Departure,
            });
        }
    }
    /// Handle a departure event
    fn handle_departure(&mut self, event_time: f64) {
        // Update statistics for time period since last event
        let time_delta = event_time - self.current_time;
        self.update_stats(time_delta);

        // Update current time and decrement customers
        self.current_time = event_time;
        self.customers_in_system -= 1;

        // If there are more customers, schedule next departure
        if self.customers_in_system > 0 {
            let departure_time = self.current_time + self.generate_service_time();
            self.event_queue.push_back(Event {
                time: departure_time,
                event_type: EventType::Departure,
            });
        }
    }
    /// Run simulation for a specified number of customers
    fn run(&mut self, total_customers_to_process: u32) {
        // Schedule first arrival
        self.schedule_initial_arrival();

        // Process events until we've served enough customers
        while self.total_customers < total_customers_to_process {
            // Sort events by time if necessary
            self.event_queue.make_contiguous();
            let mut events: Vec<_> = self.event_queue.drain(..).collect();
            events.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
            self.event_queue.extend(events);

            // Process next event
            if let Some(event) = self.event_queue.pop_front() {
                match event.event_type {
                    EventType::Arrival => self.handle_arrival(event.time),
                    EventType::Departure => self.handle_departure(event.time),
                }
            }
        }
    }

    /// Print simulation results
    fn print_results(&self) {
        println!("\nSimulation Results:");
        println!("Load (rho) = {:.3}", self.rho);
        println!("Total customers processed: {}", self.total_customers);
        println!("Simulated average customers in system: {:.3}", self.get_average_customers());
        println!("Theoretical average customers in system: {:.3}", self.get_theoretical_average());
        println!("Total simulation time: {:.3}", self.cumulative_time);
    }

    /// Run simulation and return results
    fn run_and_get_results(&mut self, total_customers_to_process: u32) -> SimulationResult {
        self.run(total_customers_to_process);
        
        SimulationResult {
            rho: self.rho,
            simulated_average: self.get_average_customers(),
            theoretical_average: self.get_theoretical_average(),
        }
    }
}
/// Function to create plot comparing theoretical and simulated results
fn create_comparison_plot(results: &[SimulationResult]) -> Result<(), Box<dyn std::error::Error>> {
    // Create plot area
    let root = BitMapBackend::new("mm1_queue_results.png", (800, 600))
        .into_drawing_area();
    
    root.fill(&WHITE)?;

    // Find max y value for plot scaling
    let max_y = results.iter()
        .map(|r| r.simulated_average.max(r.theoretical_average))
        .fold(f64::NEG_INFINITY, f64::max) * 1.1;

    // Create chart
    let mut chart = ChartBuilder::on(&root)
        .caption("M/M/1 Queue Simulation Results", ("sans-serif", 30).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..1f64, 0f64..max_y)?;

    // Configure chart
    chart
        .configure_mesh()
        .x_desc("Load (ρ)")
        .y_desc("Average Number of Customers")
        .draw()?;

    // Plot theoretical values (line)
    chart.draw_series(LineSeries::new(
        results.iter().map(|r| (r.rho, r.theoretical_average)),
        RED.filled(),
    ))?.label("Theoretical")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));

    // Plot simulated values (points)
    chart.draw_series(results.iter().map(|r| {
        Circle::new(
            (r.rho, r.simulated_average),
            5,
            BLUE.filled(),
        )
    }))?.label("Simulated")
        .legend(|(x, y)| Circle::new((x + 10, y), 5, BLUE.filled()));

    // Draw legend
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mu = 1.0;  // Fixed service rate
    let loads = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let mut results = Vec::new();

    // Run simulations for different loads
    for &rho in &loads {
        println!("\nRunning simulation for ρ = {}", rho);
        let lambda = rho * mu;
        let mut simulation = Simulation::new(lambda, mu);
        let result = simulation.run_and_get_results(10000);
        simulation.print_results();
        results.push(result);
    }

    // Create plot
    create_comparison_plot(&results)?;
    println!("\nPlot has been saved as 'mm1_queue_results.png'");

    Ok(())
}