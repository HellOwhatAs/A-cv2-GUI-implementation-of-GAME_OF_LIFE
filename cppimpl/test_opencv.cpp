#include <opencv2/core/core.hpp>//包含openCV的基本数据结构，数组操作的基本函数
#include <opencv2/highgui/highgui.hpp>//图像的交互界面，视频的捕捉也可写为#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>//图像的变换，滤波直方图，以及形状的描述等函数库
#include <opencv2/imgproc/imgproc.hpp>    
#include <opencv2/imgproc/types_c.h>   
#pragma comment( linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"" )
using namespace std;
using namespace cv;
int l = 500, windowl = 640, sight = windowl, pressing = 0, fps = 50;
bool mode = 0;
pair<int, int> sight_point(0, 0), tmp_signp(0, 0);
Mat world(l, l, CV_8UC1), new_world;
void _MouseEvent1(int x, int y) {
	sight_point.first = sight * (sight_point.first + y) / (sight + 50) - y;
	sight_point.second = sight * (sight_point.second + x) / (sight + 50) - x;
	sight_point.first = max(0, min(sight_point.first, sight-windowl));
	sight_point.second = max(0, min(sight_point.second, sight-windowl));
}
void _MouseEvent2(int x, int y) {
	sight_point.first = sight * (sight_point.first + y) / (sight - 50) - y;
	sight_point.second = sight * (sight_point.second + x) / (sight - 50) - x;
	sight_point.first = max(0, min(sight_point.first, sight - windowl));
	sight_point.second = max(0, min(sight_point.second, sight - windowl));
}
void MouseEvent(int event, int x, int y, int flags, void* param) {
	if (event == EVENT_MOUSEWHEEL) {
		if (flags < 0) {
			if (sight > windowl) {
				sight -= 50;
				_MouseEvent1(x, y);
			}
			else return;
		}
		else {
			sight += 50;
			_MouseEvent2(x, y);
		}
	}
	if (pressing) {
		sight_point.first = min(max(0, tmp_signp.first - y), sight - windowl);
		sight_point.second = min(max(0, tmp_signp.second - x), sight - windowl);

	}
	if (event == EVENT_RBUTTONDOWN) {
		tmp_signp.first = y + sight_point.first;
		tmp_signp.second = x + sight_point.second;
		pressing = 1;
	}
	else if (event == EVENT_RBUTTONUP) {
		tmp_signp.first = 0;
		tmp_signp.second = 0;
		pressing = 0;
	}
	if (mode)return;
	if (event == EVENT_LBUTTONDOWN) {
		int ptmp_x = ((y + sight_point.first) * l / (sight)), ptmp_y = ((x + sight_point.second) * l / (sight));
		if (int(world.ptr<uchar>(ptmp_x)[ptmp_y]) == 0) {
			world.ptr<uchar>(ptmp_x)[ptmp_y] = 1;
		}
		else world.ptr<uchar>(ptmp_x)[ptmp_y] = 0;
	}
}
int _f(int i, int j) {
	if(i>=0&&i<l&&j>=0&&j<l)return world.ptr(i)[j];
	return 0;
}
void update_world() {
	new_world = world.clone();
	int s0 = world.size[0], s1 = world.size[1],tmp;
	for (int i = 0; i < s0; i++) {
		for (int j = 0; j < s1; j++) {
			tmp =
				_f(i - 1, j - 1) + _f(i - 1, j) + _f(i - 1, j + 1) +
				_f(i    , j - 1) +              + _f(i    , j + 1) +
				_f(i + 1, j - 1) + _f(i + 1, j) + _f(i + 1, j + 1);
			if (tmp == 3)new_world.ptr(i)[j] = 1;
			else if (tmp < 2 || tmp>3)new_world.ptr(i)[j] = 0;
		}
	}
	world = new_world.clone();
}
pair<int, int> get_cut() {
	return {max(0,(sight_point.first * l/sight-10)),max(0,(sight_point.second*l/sight-10))};
}
pair<int, int> get_cut2() {
	return {min(l,((sight_point.first + windowl) * l/sight+10)),min(l,((sight_point.second+windowl)*l/sight+10))};
}
pair<pair<int, int>, pair<int, int>> get_cutslice(pair<int, int> cut) {
	return {
		{(sight_point.first - cut.first * double(sight) / l),(sight_point.first - cut.first * double(sight) / l) + windowl},
		{(sight_point.second - cut.second * double(sight) / l),(sight_point.second - cut.second * double(sight) / l) + windowl}
	};
}
void addagrid(Mat&tmp,pair<int, int> cut, pair<int, int> cut2, int alpha) {
	int _1 = tmp.size[0], _2 = tmp.size[1];
	int _11 = cut2.first - cut.first, _22 = cut2.second - cut.second;
	int bs = 5 * (_11 / 200);
	if (bs == 0)bs = 1;
	for (int i = cut.first % bs; i < _11; i += bs) {
		for (int j = 0; j < _2; j++) {
			tmp.ptr((int)(double(i) * _1 / _11))[j] = alpha;
		}
	}
	for (int i = cut.second % bs; i < _22; i += bs) {
		for (int j = 0; j < _1; j++) {
			tmp.ptr(j)[(int)((double(i) * _2) / _22)] = alpha;
		}
	}
}
int main(char argc, char* argv[]) {
	bool flag;
	int k;
	Mat tmp;
	pair<pair<int, int>, pair<int, int>> stmp;
	world = 0;
	namedWindow("game of life", WINDOW_AUTOSIZE);
	setMouseCallback("game of life", MouseEvent);
	while (1) {
		flag = 1;
		while (1) {
			pair<int,int> cut = get_cut(), cut2 = get_cut2();
			resize(world(Rect(
				cut.second, 
				cut.first, 
				cut2.second- cut.second,
				cut2.first- cut.first
			))*255,tmp,Size(0,0),double(sight)/l,double(sight)/l,INTER_NEAREST);
			addagrid(tmp, cut, cut2, 50);
			stmp = get_cutslice(cut);
			imshow("game of life", tmp(Rect(
				stmp.second.first,
				stmp.first.first,
				stmp.second.second- stmp.second.first,
				stmp.first.second- stmp.first.first
			)));
			if (mode) {
				k = waitKey(fps);
				if (k == 61 && fps > 2)--fps;
				else if (k == 45)++fps;
			}
			else k = waitKey(100);
			if (k == 27 || getWindowProperty("game of life", WND_PROP_VISIBLE) < 1.0) {
				flag = 0;
				break;
			}
			if (k == 32)break;
			if (mode)update_world();
		}
		if (!flag)break;
		mode = !mode;
	}
	destroyAllWindows();
	return 0;
}

