#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
#define WIDTH 800
#define HEIGHT 600

Mat image;
int xRacket;
int yRacket;
int xBall;
int yBall;
int xFood;
int yFood;
int points = 0;
double xBallSpeed = 10;
double yBallSpeed = 10;
int speed = 10;
int step = 10;
bool firstrun = true;

void redraw() {
	rectangle(image, Point(0, 0), Point(WIDTH, HEIGHT), Scalar(0, 0, 0), FILLED);															// Background
	rectangle(image, Point(xRacket, yRacket), Point(xRacket + 200, yRacket + 20), Scalar(255, 166, 26), FILLED);							// Racket
	rectangle(image, Point(xFood, yFood), Point(xFood + 50, yFood + 50), Scalar(255, 50, 50), FILLED);										// Food
	circle(image, Point(xBall, yBall), 5, Scalar(220, 220, 220), FILLED);																	// Ball

	rectangle(image, Point(0, 0), Point(20, HEIGHT), Scalar(100, 200, 100), FILLED);														// Sides
	rectangle(image, Point(0, 0), Point(WIDTH, 20), Scalar(100, 200, 100), FILLED);
	rectangle(image, Point(WIDTH-20, 0), Point(WIDTH, HEIGHT), Scalar(100, 200, 100), FILLED);
	putText(image, "Points: " + to_string(points), Point(20, 40), FONT_HERSHEY_DUPLEX, 0.75, CV_RGB(220, 220, 220), 1);						// Points counter
	
	if (firstrun) // Press any key to start text
	{
		putText(image, "Press any key to start", Point(WIDTH / 2 - 180, HEIGHT / 2 - 30), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(220, 220, 220), 2);
		firstrun = false;
	}
	imshow("Display window", image);
}

void setFoodPosition() // Replace food on screen
{
	srand(time(NULL));
	xFood = rand() % 700 + 50;
	yFood = rand() % 500 + 20;
}

void initAngle() // Set the initial trajectory of ball
{
	srand(time(NULL)); 
	xBallSpeed = rand() % 20 + 1;
	yBallSpeed = 20 - xBallSpeed;

	if (rand() % 10 > 5) // Invert the x direction randomly
	{
		xBallSpeed = xBallSpeed * (-1);
	}
	if (rand() % 10 <= 5) // Invert the y direction randomly
	{
		yBallSpeed = yBallSpeed * (-1);
	}
}

void checkIfDropped() // Check if ball has crossed the bottom of the screen
{
	if (yBall >= 600)
	{
		putText(image, "YOU LOSE", Point(WIDTH / 2 - 80, HEIGHT / 2 - 30), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 0, 0), 2);
		imshow("Display window", image);
		waitKey(0);
		exit(0);
	}
}

void applyBallPosition() // Calculate the velocity
{
	if (xBall >= 0 && xBall <= 20) // Left wall
	{
		xBallSpeed = xBallSpeed * (-1);
		xBall = 20;
	}
	else if (xBall >= WIDTH - 20 && xBall < WIDTH) // Right wall
	{
		xBallSpeed = xBallSpeed * (-1);
		xBall = WIDTH - 20;
	}
	else if (yBall >= 0 && yBall < 20) // Top wall 
	{ 
		yBallSpeed = yBallSpeed * (-1);
		yBall = 20;
	}
	else if ((yBall >= 550 && yBall <= 570) && (xRacket <= xBall && xRacket + 100 >= xBall)) // Racket left side
	{
		double px = xBall - xRacket;
		yBallSpeed = abs(20 * (px / 100)) * (-1);
		xBallSpeed = abs(20 + yBallSpeed) * (-1);

		if (yBallSpeed <= 0 && yBallSpeed > -2) // Racket left edge
		{
			yBallSpeed = -2;
			xBallSpeed = -18;
		}
	}
	else if ((yBall >= 550 && yBall <= 570) && (xRacket + 100 < xBall && xRacket + 200 >= xBall)) // Racket right side
	{
		double px = xBall - (xRacket + 100);
		xBallSpeed = abs(20 * (px / 100));
		yBallSpeed = abs(20 - xBallSpeed) * (-1);

		if (yBallSpeed <= 0 && yBallSpeed > -2) // Racket right edge
		{
			yBallSpeed = -2;
			xBallSpeed = 18;
		}
	}
	// Apply next position on map 
	xBall += xBallSpeed;
	yBall += yBallSpeed;
}

void checkFoodHit() 
{
	if (xBall >= xFood && xBall <= xFood+50 && yBall <= yFood+50 && yBall > yFood) // Middle of ball is inside the food square
	{
		points++;
		setFoodPosition();
	}
	if (points == 5) // Winner screen
	{
		putText(image, "YOU WIN", Point(WIDTH / 2 - 80, HEIGHT / 2 - 30), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 255, 0), 2);
		imshow("Display window", image);
		waitKey(0);
		exit(0);
	}
}

int main()
{
	// Set initial params
	image = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

	xRacket = WIDTH / 2 - 100;
	yRacket = HEIGHT - 50;

	xBall = WIDTH / 2;
	yBall = HEIGHT / 2;

	setFoodPosition();
	initAngle();
	redraw();

	namedWindow("Display window", WINDOW_AUTOSIZE);
	imshow("Display window", image);
	waitKey(0);

	int key;
	while (true)
	{
		checkIfDropped();
		checkFoodHit();
		applyBallPosition();

		// Controls
		key = waitKey(30);
		switch (key)
		{
		case 'a':
			if (xRacket >= 20) 
			{
				xRacket -= step;
			}
			break;

		case 'd':
			if (xRacket + 200 != WIDTH-20)
			{
				xRacket += step;
			}
			break;
		}
		redraw();
	}
	return 0;
}