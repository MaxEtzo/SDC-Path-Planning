#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "helper_functions.h"
#include "spline.h"

using namespace std;

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
	float ref_vel = 0.0;
	float ref_acc = 0.0;
	float max_vel = 49.0 / 2.24;
	float max_acc = 9.0;
	float max_jerk = 9.0;
	h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy,&max_s,&target_lane,&ref_vel,&ref_acc,&max_vel,&max_acc,&max_jerk](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
				uWS::OpCode opCode) {
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
			double car_yaw = j[1]["yaw"];
			double car_speed = j[1]["speed"];
			int car_lane = floor(car_d / 4.0);

			// Previous path data given to the Planner
			auto previous_path_x = j[1]["previous_path_x"];
			auto previous_path_y = j[1]["previous_path_y"];
			// Previous path's end s and d values 
			double end_path_s = j[1]["end_path_s"];
			double end_path_d = j[1]["end_path_d"];
			int prev_size = previous_path_x.size();

			// Sensor Fusion Data, a list of all other cars on the same side of the road.
			auto sensor_fusion = j[1]["sensor_fusion"];
			json msgJson;
			// collect information about the traffic on all lanes
			float s_horizon = 50.0;
			float freespace_ahead[3] = {s_horizon, s_horizon, s_horizon};
			float freespace_behind[3] = {-s_horizon, -s_horizon, -s_horizon};
			float speed_ahead[3] = {max_vel, max_vel, max_vel};
			float speed_behind[3] = {0.0, 0.0, 0.0};
			for (int i = 0; i < sensor_fusion.size(); i++)
			{
				// cars to be checked would be called "check_car"
				float vx = sensor_fusion[i][3];
				float vy = sensor_fusion[i][4];
				float check_car_vel = sqrt(vx*vx + vy*vy);
				float check_car_s = sensor_fusion[i][5];
				float check_car_d = sensor_fusion[i][6];
				int check_lane = floor(check_car_d / 4.0);
				// distance difference between ego and check car
				float delta_s = check_car_s - car_s;
				if (delta_s < -max_s + s_horizon) // deal with singularity at 0.
					delta_s += max_s;
				if (delta_s > max_s - s_horizon)
					delta_s -= max_s;
				if (delta_s < freespace_ahead[check_lane] && delta_s > 0)
				{
					freespace_ahead[check_lane] = delta_s;
					speed_ahead[check_lane] = check_car_vel;	
				}	
				else if (delta_s > freespace_behind[check_lane] && delta_s <= 0)
				{
					freespace_behind[check_lane] = delta_s;
					speed_behind[check_lane] = check_car_vel;
				}
			}
			// calculate costs associated with each possible state
			// states: Lane Change Left; Keep Lane; Lane Change Right in the same order	
			// state = -1 represents LCL, 0 - KL, and 1 - LCR
			float costs[3] = {1E6, 1E6, 1E6};
			for (int i = 0; i < 3; i++) // iterate through each lane
			{
				float cost = (1 - exp(-(s_horizon - freespace_ahead[i])/s_horizon)) + (1 - exp(-(max_vel - speed_ahead[i])/max_vel)); 	
				int state = max(min(i - target_lane, 1), -1); // no double lane change, but overall direction towards lower cost must be preserved
				if (cost < costs[state + 1])
					costs[state + 1] = cost + abs(state) * 0.05; // prefer keep lane over lane chage	
			}	

			// now prediction phases come. prepare some time horizons!
			float t_hor = 50 * 0.02;
			float t2 = t_hor * t_hor;
			float t3 = t2 * t_hor;
			// find minimum cost
			float min_cost = 1E6;
			float safety_buffer = 10;
			int state = 0;
			for (int i = 0; i < 3; i++) // iterate through states this time
			{
				int next_lane = target_lane + i - 1;
				// given current ego position, velocity and acceleration 
				// predict whether merge can happen
				if (i != 1)
				{
					float projection_ahead = freespace_ahead[next_lane] + (speed_ahead[next_lane] - ref_vel)*t_hor - 0.5 * ref_acc*t2;
					float projection_behind = freespace_behind[next_lane] + (speed_behind[next_lane] - ref_vel)*t_hor - 0.5*ref_acc*t2;
					if (freespace_ahead[next_lane] < safety_buffer || projection_ahead < safety_buffer)
						costs[i] += 100.0;
					else if (freespace_behind[next_lane] > -safety_buffer || projection_behind > -safety_buffer)
						costs[i] += 100.0;					
				}
				if (costs[i] < min_cost)
				{
					min_cost = costs[i];
					state = i - 1;
				}
				std::cout << "cost for state " << i - 1 << " " << costs[i] << std::endl;
			}

			// check if previous maneuver is completed (e.g. change lane or recovery to lane center)
			float eps = 0.2;
			if (fabs(car_d - 4*target_lane - 2) < eps)
			{
				target_lane += state;
			}

			// control jerk, acceleration and speed to avoid accidents
			float dt = 0.05; // empirically measured
			float projection_ahead = freespace_ahead[car_lane] + (speed_ahead[car_lane] - ref_vel) * t_hor  - 0.5 * ref_acc * t2;
			float projection_vel = ref_vel + ref_acc * t_hor;
			float jerk = 0.0;
			if (projection_ahead < safety_buffer)
			{
				jerk = max(6 * (projection_ahead - safety_buffer) / t3, -max_jerk);
			}
			else if (projection_vel > max_vel) 
			{
				jerk = max(2 * (max_vel - projection_vel) / t2, -max_jerk);
			}
			else if (projection_vel < max_vel && ref_acc < max_acc)
			{
				jerk = min(2 * (max_vel - projection_vel) / t2, max_jerk);
			}
			ref_acc = max(min(ref_acc + jerk * dt, max_acc), -max_acc);
			ref_vel = max(min(ref_vel + ref_acc * dt, max_vel), 0.0f); // no reverse movement on the highway
			std::cout << "current jerk: " << jerk << std::endl;
			std::cout << "current acceleration: " << ref_acc << std::endl;	
			std::cout << "current velocity: " << ref_vel << std::endl;

			// trajectory planning starts here
			vector<double> ptsx;
			vector<double> ptsy;

			double ref_x = car_x;
			double ref_y = car_y;
			double ref_yaw = deg2rad(car_yaw);
			double ref_x_prev;
			double ref_y_prev;
			if (prev_size < 2)// no previous path, use two points tangent to the car
			{
				double ref_x_prev = ref_x - cos(ref_yaw);
				double ref_y_prev = ref_y - sin(ref_yaw);
			}
			else // use first two previous paths point for smooth transition, and yet to reflect latest controls (jerk, acc, vel)
			{
				ref_x = previous_path_x[1];
				ref_y = previous_path_y[1];
				ref_x_prev = previous_path_x[0];
				ref_y_prev = previous_path_y[0];
				ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);
			}

			ptsx.push_back(ref_x_prev);
			ptsx.push_back(ref_x);
			ptsy.push_back(ref_y_prev);
			ptsy.push_back(ref_y);
			
			vector<double> next_wp0 = getXY(car_s + 40, (2+4*target_lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
			vector<double> next_wp1 = getXY(car_s + 70, (2+4*target_lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
			vector<double> next_wp2 = getXY(car_s + 100, (2+4*target_lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
			
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

			vector<double> next_x_vals;
			vector<double> next_y_vals;

			// calculate how to break up spline points so that we travel at our desired reference velocity
			double target_x = 30.0;
			double target_y = s(target_x);
			double target_dist = sqrt(target_x*target_x + target_y*target_y);
			double N = target_dist / (0.02*ref_vel);
			double x_last = 0;
			double y_last = 0;
			if (prev_size > 1)
			{
				next_x_vals.push_back(previous_path_x[0]);
				next_x_vals.push_back(previous_path_x[1]);
				next_y_vals.push_back(previous_path_y[0]);
				next_y_vals.push_back(previous_path_y[1]);
			}
			for (int i = 0; i < 50; i++)
			{
				double x_point = x_last + (target_x)/N;
				double y_point = s(x_point);
				// points above do not guarantee speed limit compliance!
				
				double yaw = atan2(y_point - y_last, x_point - x_last);
				x_point = x_last + cos(yaw) * ref_vel * 0.02;
				y_point = y_last + sin(yaw) * ref_vel * 0.02;
				x_last = x_point;
				y_last = y_point;

				// translate back to global coordinate system
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

