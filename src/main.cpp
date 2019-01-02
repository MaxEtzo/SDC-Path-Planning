#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "helper_functions.h"
#include "spline.h"

#define MAX_VEL_ 49.0/2.24
using namespace std;
using hires_clock = chrono::high_resolution_clock;
using usec = chrono::microseconds;

// for convenience
using json = nlohmann::json;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
	auto found_null = s.find("null");
	auto b1 = s.find_first_of("[");
	auto b2 = s.find_first_of("}");
	if (found_null != string::npos) {
		return "";
	} else if (b1 != string::npos && b2 != string::npos) {
		return s.substr(b1, b2 - b1 + 2);
	}
	return "";
}


int main() {
	uWS::Hub h;

	// Load up map values for waypoint's x,y,s and d normalized normal vectors
	vector<double> map_waypoints_x;
	vector<double> map_waypoints_y;
	vector<double> map_waypoints_s;
	vector<double> map_waypoints_dx;
	vector<double> map_waypoints_dy;

	// Waypoint map to read from
	string map_file_ = "../data/highway_map.csv";
	// The max s value before wrapping around the track back to 0
	double max_s = 6945.554;

	ifstream in_map_(map_file_.c_str(), ifstream::in);

	string line;
	while (getline(in_map_, line)) {
		istringstream iss(line);
		double x;
		double y;
		float s;
		float d_x;
		float d_y;
		iss >> x;
		iss >> y;
		iss >> s;
		iss >> d_x;
		iss >> d_y;
		map_waypoints_x.push_back(x);
		map_waypoints_y.push_back(y);
		map_waypoints_s.push_back(s);
		map_waypoints_dx.push_back(d_x);
		map_waypoints_dy.push_back(d_y);
	}

	int target_lane = 1;
	int next_lane = 1;
	double ref_vel = 0.0;
	double ref_acc = 0.0;
	double ref_jerk = 0.0;
	double max_vel = MAX_VEL_;
	double max_acc = 6.0; // only limits tangential acceleration
	double max_jerk = 6.0; // only limits tangential jerk
	string state = "KL";
	h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy,&max_s,&target_lane,&next_lane,&ref_vel,&ref_acc,&ref_jerk,&max_vel,&max_acc,&max_jerk,&state](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
			// "42" at the start of the message means there's a websocket message event.
			// The 4 signifies a websocket message
			// The 2 signifies a websocket event
			if (length && length > 2 && data[0] == '4' && data[1] == '2') {

			auto s = hasData(data);

			if (s != "") {
			auto j = json::parse(s);

			string event = j[0].get<string>();

			if (event == "telemetry") {
			// j[1] is the data JSON object

			// Main car's localization Data
			double car_x = j[1]["x"];
			double car_y = j[1]["y"];
			double car_s = j[1]["s"];
			double car_d = j[1]["d"];
			double car_yaw = deg2rad(j[1]["yaw"]);
			double car_speed = double(j[1]["speed"]) / 2.24;
			int car_lane = floor(car_d / 4.0);

			// Previous path data given to the Planner
			auto previous_path_x = j[1]["previous_path_x"];
			auto previous_path_y = j[1]["previous_path_y"];
			// Previous path's end s and d values 
			double end_path_s = j[1]["end_path_s"];
			double end_path_d = j[1]["end_path_d"];
			int prev_size = previous_path_x.size();

			// dt since last cycle
			double dt = (50 - prev_size)*0.02;
			// estimate current reference velocity and acceleration 
			ref_acc = max(min(ref_acc + ref_jerk * dt, max_acc), -max_acc);
			if (prev_size >= 2)
			{
				double next_delta_x = double(previous_path_x[1]) - double(previous_path_x[0]);
				double next_delta_y = double(previous_path_y[1]) - double(previous_path_y[0]);
				ref_vel = sqrt(next_delta_x*next_delta_x + next_delta_y*next_delta_y)/0.02;
			}
			else ref_vel = max(min(ref_vel + ref_acc * dt + 0.5 * ref_jerk * dt * dt, max_vel), 0.0);

			// Sensor Fusion Data, a list of all other cars on the same side of the road.
			auto sensor_fusion = j[1]["sensor_fusion"];
			json msgJson;
			// collect information about the traffic on all lanes
			/*
			 * Sensor Fusion related code 
			 * Identify the closest cars ahead and behind in each lane
			 * Predict if car is changing lane. If yes, it limits both current lane and target lane
			 */
			double s_horizon = 40.0;
			double freespace_ahead[3] = {s_horizon, s_horizon, s_horizon};
			double freespace_behind[3] = {-s_horizon, -s_horizon, -s_horizon};
			double speed_ahead[3] = {max_vel, max_vel, max_vel};
			double speed_behind[3] = {0.0, 0.0, 0.0};
			for (int i = 0; i < sensor_fusion.size(); i++)
			{
				// cars to be checked would be called "check"
				double x = sensor_fusion[i][1];
				double y = sensor_fusion[i][2];
				double vx = sensor_fusion[i][3];
				double vy = sensor_fusion[i][4];
				double check_s = sensor_fusion[i][5];
				double check_d = sensor_fusion[i][6];
				int check_lane = floor(check_d / 4.0);
				if (check_d < 0) 
					continue;

				vector<double> frenetVel = getFrenetVelocity(x, y, vx, vy, map_waypoints_x, map_waypoints_y); 	
				double check_s_dot = frenetVel[0];
				double check_d_dot = frenetVel[1];

				// distance difference between ego and check car
				double delta_s = check_s - car_s;
				// deal with singularity at max_s = 0.
				if (fabs(delta_s) >= max_s - s_horizon)
					delta_s += (delta_s > 0 ? -1 : 1)*max_s;

				// car poses space and velocity limits on both current lane and target lane
				if (fabs(delta_s) < s_horizon)
				{
					// predict cars target lane, i.e. any lane change maneuver taking place
					int check_target_lane = check_lane;
					double d_offset = check_d-4*check_lane-2;
					double p = 1.0/(1+exp(-d_offset))+1.0/(1+exp(-check_d_dot))-1;
					if (p > 0.65)
					{
						check_target_lane++;	
					}
					else if (p < -0.65)
					{
						check_target_lane--;
					}

					if (delta_s < freespace_ahead[check_lane] && delta_s > 0)
					{
						freespace_ahead[check_lane] = delta_s;
						speed_ahead[check_lane] = check_s_dot;
					}	
					else if (delta_s > freespace_behind[check_lane] && delta_s <= 0)
					{
						freespace_behind[check_lane] = delta_s;
						speed_behind[check_lane] = check_s_dot;
					}
					if (delta_s < freespace_ahead[check_target_lane] && delta_s > 0)
					{
						freespace_ahead[check_target_lane] = delta_s;
						speed_ahead[check_target_lane] = check_s_dot;	
					}	
					else if (delta_s > freespace_behind[check_target_lane] && delta_s <= 0)
					{
						freespace_behind[check_target_lane] = delta_s;
						speed_behind[check_target_lane] = check_s_dot;
					}
				}
			}

			/*
			 * Calculate costs for each lane based on freespace and speed ahead 
			 * Identify direction towards lower cost lane
			 * Directions: -1 - Left, 0 - Forward, 1 - Right	
			 */
			double costs[3] = {1E6, 1E6, 1E6};
			double min_cost = 1E6;
			int target_lane = car_lane;
			for (int i = 0; i < 3; i++) // iterate through each lane
			{
				costs[i] = (1 - exp(-(s_horizon - freespace_ahead[i])/s_horizon)) + (1 - exp(-(max_vel - speed_ahead[i])/max_vel)); 	
				// prefer KEEP LANE
				if (i == car_lane)
					costs[i] -= 0.1;

				if (costs[i] < min_cost)
				{
					min_cost = costs[i];
					target_lane = i;
				}
			}
			int direction = max(min(target_lane - car_lane, 1), -1);

			// Time horizon and its powers
			double t_hor = 50 * 0.02;
			double t2 = t_hor * t_hor;
			double t3 = t2 * t_hor;

			/*
			 * Based on the direction either keep lane, prepare lane change or execute lane change
			 */
			double target_vel = max_vel; // target velocity for ego car
			double safety_buffer = 20; // default safety_buffer for KEEP LANE
			double safety_buffer_LC = 10; // safety_buffer for LANE CHANGE
			double ref_lane_space = freespace_ahead[car_lane]; // reference lane space ahead
			double ref_lane_speed = speed_ahead[car_lane]; // reference lane speed ahead

			// buffer_ahead and buffer_behind are needed for PLC and LC
			int intended_lane = car_lane + direction;
			double buffer_ahead = min(freespace_ahead[intended_lane], freespace_ahead[car_lane]);
			double buffer_behind = max(freespace_behind[intended_lane], freespace_behind[car_lane]);

			if (state.compare("LC") != 0) // only if Lane Change is not in progress
			{
				next_lane = car_lane;
				if (direction == 0)
				{
					state = "KL";
				} 
				else // Prepare Lane Change   
				{
					if (buffer_ahead - buffer_behind > 2*safety_buffer_LC) // lane change possible -> PREPARE
					{
						state = "PLC";
						ref_lane_space = buffer_ahead;
						if (freespace_ahead[intended_lane] < freespace_ahead[car_lane] - 0.3)
						{
							ref_lane_speed = speed_ahead[intended_lane]; 	
							target_vel = ref_lane_speed - 1.0; // decelerate to merge into intended_lane
						}
						else if (freespace_ahead[car_lane] < freespace_ahead[intended_lane] - 0.3)
							ref_lane_speed = speed_ahead[car_lane]; // limited by current lane's speed
						else
							ref_lane_speed = min(speed_ahead[car_lane], speed_ahead[intended_lane]); // choose minimum of two
					}
					else 
						state = "KL";
				}
			}
			else if (car_lane != next_lane)// ongoing Lane Change; set correct ref_lane_space and ref_lane_speed 
			{
				double cur_buffer_ahead = min(freespace_ahead[car_lane], freespace_ahead[next_lane]); // based on next_lane and not direction (in case plan changed)	
				ref_lane_space = cur_buffer_ahead;
				if (freespace_ahead[next_lane] < freespace_ahead[car_lane] - 0.5)
					ref_lane_speed = speed_ahead[next_lane];	
				else if (freespace_ahead[car_lane] < freespace_ahead[next_lane] - 0.5)
					ref_lane_speed = speed_ahead[car_lane];
				else
					ref_lane_speed = min(speed_ahead[car_lane], speed_ahead[next_lane]);
			}
			if (state.compare("PLC") == 0) // Lane change only after PLC 
			{
				if (buffer_ahead > safety_buffer_LC && buffer_behind < -safety_buffer_LC) // check if change lane can be initiated
				{
					state = "LC";
					next_lane = car_lane + direction;
					// printing out
					string dir = (direction > 0 ? "Right" : "Left");
					cout << "Lane Change " << dir << " INITIATED" << endl;
					cout << "Target Lane " << target_lane << endl;
					cout << "Next Lane " << next_lane << endl;
				}
			}

			if (state.compare("PLC") == 0 || state.compare("LC") == 0)
				safety_buffer = safety_buffer_LC;

			/*
			 * Control Jerk, acceleration and speed!
			 */
			double projection_space = ref_lane_space + (ref_lane_speed - ref_vel) * t_hor  - 0.5 * ref_acc * t2;
			double projection_speed = ref_vel + ref_acc * t_hor;
			if (projection_space < safety_buffer)
			{
				// for safety max_jerk can be used
				ref_jerk = max(min(6 * (projection_space - safety_buffer) / t3, max_jerk), -max_jerk);
			}
			else // use reduced jerk;
			{
				double red_jerk = max_jerk * 1.0;
				ref_jerk = max(min(2 * (target_vel - projection_speed) / t2, red_jerk), -red_jerk);
			}

			// check if lane change completed (update state to KL)
			double eps = 0.7;
			if (state.compare("LC") == 0 && fabs(car_d - 4*next_lane - 2) < eps) 
			{
				state = "KL";
				cout << "Lane Change COMPLETED" << endl;
				cout << "Current Lane " << car_lane << endl;
				cout << "Target Lane " << target_lane << endl;
			} 
			/* 
			 * trajectory planning starts here
			 */
			vector<double> next_x_vals;
			vector<double> next_y_vals;
			vector<double> ptsx;
			vector<double> ptsy;

			double ref_x = car_x;
			double ref_y = car_y;
			double ref_yaw = car_yaw;

			if (prev_size < 2) // no previous path, use two points tangent to the car
			{
				ptsx.push_back(ref_x - cos(ref_yaw));
				ptsx.push_back(ref_x);
				ptsy.push_back(ref_y - sin(ref_yaw));
				ptsy.push_back(ref_y);
				prev_size = 0;
			}
			else // use only first two points for smooth transition (for spline), and first point for next values
			{
				ptsx.push_back(previous_path_x[0]);
				ptsx.push_back(previous_path_x[1]);
				ptsy.push_back(previous_path_y[0]);
				ptsy.push_back(previous_path_y[1]);
				next_x_vals.push_back(ptsx[0]);
				next_y_vals.push_back(ptsy[0]);
				ref_x = ptsx[0];
				ref_y = ptsy[0];
				ref_yaw = atan2(ref_y - car_y, ref_x - car_x);
				prev_size = 1;
			}

			double delta_d = -0.2 * (next_lane - 1);// slightly avoid road boundaries (easy to catch outside of lane error due to interpolation)
			vector<double> next_wp0 = getXY(car_s + 30, (2+4*next_lane)+delta_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
			vector<double> next_wp1 = getXY(car_s + 60, (2+4*next_lane)+delta_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
			vector<double> next_wp2 = getXY(car_s + 90, (2+4*next_lane)+delta_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);

			ptsx.push_back(next_wp0[0]);
			ptsx.push_back(next_wp1[0]);
			ptsx.push_back(next_wp2[0]);

			ptsy.push_back(next_wp0[1]);
			ptsy.push_back(next_wp1[1]);
			ptsy.push_back(next_wp2[1]);

			//transfrom into a cars reference
			for (int i = 0; i < ptsx.size(); i++)
			{
				double shift_x = ptsx[i] - ref_x;
				double shift_y = ptsy[i] - ref_y;

				ptsx[i] = shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw);
				ptsy[i] = shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw);
			}

			// create a spline
			tk::spline s;
			s.set_points(ptsx, ptsy);

			// calculate how to break up spline points so that we travel at our desired reference velocity
			double accel = ref_acc; // max(min(ref_acc + ref_jerk * 0.02, max_acc), -max_acc);
			double speed = ref_vel;

			double target_x = 30.0;
			double target_y = s(target_x);
			double x_last = 0;
			double y_last = 0;

			for (int i = prev_size; i < 50; i++)
			{
				double delta_x = target_x - x_last;
				double delta_y = target_y - y_last;	
				double target_dist = sqrt(delta_x*delta_x + delta_y*delta_y);
				double N = target_dist / (0.02*speed);
				double x_point = x_last + (delta_x)/N;
				double y_point = s(x_point);
				// points above do not guarantee speed limit compliance!
				double yaw = atan2(y_point - y_last, x_point - x_last);
				x_point = x_last + cos(yaw) * speed * 0.02;
				y_point = y_last + sin(yaw) * speed * 0.02;
				x_last = x_point;
				y_last = y_point;
				// update speed and accel
				accel = max(min(accel + ref_jerk * 0.02, max_acc), -max_acc);
				speed = max(min(speed + accel * 0.02, max_vel), 0.0);

				// translate back to global coordinate systeref
				x_point = x_last * cos(ref_yaw) - y_last * sin(ref_yaw) + ref_x;
				y_point = x_last * sin(ref_yaw) + y_last * cos(ref_yaw) + ref_y;

				next_x_vals.push_back(x_point);
				next_y_vals.push_back(y_point);
			}


			// TODO: define a path made up of (x,y) points that the car will visit sequentially every .02 seconds
			msgJson["next_x"] = next_x_vals;
			msgJson["next_y"] = next_y_vals;

			auto msg = "42[\"control\","+ msgJson.dump()+"]";

			//this_thread::sleep_for(chrono::milliseconds(1000));
			ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);

			}
			} else {
				// Manual driving
				std::string msg = "42[\"manual\",{}]";
				ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
			}
			}
	});

	// We don't need this since we're not using HTTP but if it's removed the
	// program
	// doesn't compile :-(
	h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
				size_t, size_t) {
			const std::string s = "<h1>Hello world!</h1>";
			if (req.getUrl().valueLength == 1) {
			res->end(s.data(), s.length());
			} else {
			// i guess this should be done more gracefully?
			res->end(nullptr, 0);
			}
			});

	h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
			std::cout << "Connected!!!" << std::endl;
			});

	h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
				char *message, size_t length) {
			ws.close();
			std::cout << "Disconnected" << std::endl;
			});

	int port = 4567;
	if (h.listen(port)) {
		std::cout << "Listening to port " << port << std::endl;
	} else {
		std::cerr << "Failed to listen to port" << std::endl;
		return -1;
	}
	h.run();
}
