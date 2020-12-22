#ifndef sgl_vec
#define sgl_vec

#include <math.h>
#include <stdio.h>

class vecd2;
class vecd3;
class vecd4;
class veci2;
class veci3;
class veci4;

class quaternion;

class vecd2 {
public:
	double x;
	double y;
	
	vecd2();
	vecd2(double x, double y);
	
	// Returns the magnitude of this vector
	double mag();
	// Returns the squared magnitude of this vector
	double sqr_mag();
	
	vecd2 operator+(const vecd2& a) const;
	vecd2 operator-(const vecd2& a) const;
	vecd2 operator*(const vecd2& a) const;
	vecd2 operator/(const vecd2& a) const;
	vecd2 operator*(double a) const;
	vecd2 operator/(double a) const;
	vecd2 operator-();
	
	bool operator==(vecd2 a);
	
	// Returns whether this vector is nan. Only returns true if all elements are nan
	bool is_nan();
	
	// Returns a normalized version of this vector
	vecd2 normalize();
	vecd2 normalize(double t);
	
	// Returns the Dot Product of two vectors
	static double dot(vecd2 a, vecd2 b);	
	
	static vecd2 lerp(vecd2 a, vecd2 b, float t);
	static vecd2 bez3(vecd2 s, vecd2 c, vecd2 d, float t);
};

class veci2 {
public:
	int x;
	int y;
	
	veci2();
	veci2(int x, int y);
	
	// Returns the magnitude of this vector
	double mag();
	// Returns the square magnitude of this vector
	int sqr_mag();
	
	veci2 operator+(const veci2& a) const;
	veci2 operator-(const veci2& a) const;
	veci2 operator*(const veci2& a) const;
	veci2 operator/(const veci2& a) const;
	veci2 operator*(int a) const;
	veci2 operator/(int a) const;
	veci2 operator-();
	
	bool operator==(veci2 a);
	
	// Returns a normalized version of this vector
	vecd2 normalize();
	vecd2 normalize(double t);
	
	static int dot(vecd2 a, vecd2 b);	
};

class vecd3 {
public:
	double x;
	double y;
	double z;
	
	vecd3();
	vecd3(double x, double y, double z);
	
	double mag();
	double sqr_mag();
	
	vecd3 operator+(const vecd3& a) const;
	vecd3 operator-(const vecd3& a) const;
	vecd3 operator*(const vecd3& a) const;
	vecd3 operator/(const vecd3& a) const;
	vecd3 operator*(double a) const;
	vecd3 operator/(double a) const;
	vecd3 operator-();
	
	bool operator==(vecd3 a);
	
	// Returns whether this vector is nan. Only returns true if all elements are nan
	bool is_nan();
	
	vecd3 normalize();
	vecd3 normalize(double t);
	
	static double dot(vecd3 a, vecd3 b);
	static vecd3 cross(vecd3 a, vecd3 b);
	
	static vecd3 lerp(vecd3 a, vecd3 b, float t);
	static vecd3 bez3(vecd3 s, vecd3 c, vecd3 d, float t);
};

class veci3 {
public:
	int x;
	int y;
	int z;
	
	veci3();
	veci3(int x, int y, int z);
	
	double mag();
	int sqr_mag();
	
	veci3 operator+(const veci3& a) const;
	veci3 operator-(const veci3& a) const;
	veci3 operator*(const veci3& a) const;
	veci3 operator/(const veci3& a) const;
	veci3 operator*(int a) const;
	veci3 operator/(int a) const;
	veci3 operator-();
	
	bool operator==(veci3 a);
	
	vecd3 normalize();
	vecd3 normalize(double t);
	
	static int dot(veci3 a, veci3 b);
	static veci3 cross(veci3 a, veci3 b);
};

class vecd4 {
public:
	double w;
	double x;
	double y;
	double z;
	
	vecd4();
	vecd4(double w, double x, double y, double z);
	
	double mag();
	double sqr_mag();
	
	vecd4 operator+(const vecd4& a) const;
	vecd4 operator-(const vecd4& a) const;
	vecd4 operator*(const vecd4& a) const;
	vecd4 operator/(const vecd4& a) const;
	vecd4 operator*(double a) const;
	vecd4 operator/(double a) const;
	vecd4 operator-();
	
	bool operator==(vecd4 a);
	
	// Returns whether this vector is nan. Only returns true if all elements are nan
	bool is_nan();
	
	vecd4 normalize();
	vecd4 normalize(double t);
	
	static double dot(vecd4 a, vecd4 b);
	
	static vecd4 lerp(vecd4 a, vecd4 b, float t);
	static vecd4 bez3(vecd4 s, vecd4 c, vecd4 d, float t);
};

class veci4 {
public:
	int w;
	int x;
	int y;
	int z;
	
	veci4();
	veci4(int w, int x, int y, int z);
	
	double mag();
	int sqr_mag();
	
	veci4 operator+(const veci4& a) const;
	veci4 operator-(const veci4& a) const;
	veci4 operator*(const veci4& a) const;
	veci4 operator/(const veci4& a) const;
	veci4 operator*(int a) const;
	veci4 operator/(int a) const;
	veci4 operator-();
	
	bool operator==(veci4 a);
	
	vecd4 normalize();
	vecd4 normalize(double t);
	
	static int dot(veci4 a, veci4 b);
};

class quaternion : public vecd4 {
public:
	quaternion();
	quaternion(double w, double x, double y, double z);
	quaternion(vecd3 axis, double theta);
	
	quaternion operator+(const quaternion& a) const;
	quaternion operator-(const quaternion& a) const;
	quaternion operator*(const quaternion& a) const;
	quaternion operator/(const quaternion& a) const;
	quaternion operator*(double a) const;
	quaternion operator/(double a) const;
	quaternion operator-() const;
	
	quaternion operator=(vecd4) const;
	
	bool operator==(veci4 a);
	
	quaternion operator!() const;
	
	// Normalize identical to the vecd4 normalize. This version exists so that quaternion::normalize() will return a quaternion.
	quaternion normalize();
	quaternion normalize(double t);
	
	static quaternion hamilton(const quaternion& a, const quaternion& b);
	static vecd3 vhamilton(const quaternion& a, const quaternion& b);
	
	quaternion& mhamilton(quaternion& a, const quaternion& b);
	
	vecd3 apply(const vecd3& in) const;
	
	static vecd3 rotate(vecd3 in, vecd3 axis_offset, vecd3 axis_dir, double theta);
	static vecd3 rotate(vecd3 in, vecd3 axis_offset, quaternion r);
	
	static quaternion slerp(const quaternion& a, const quaternion& b, double t); 
	
	static quaternion bez3(const quaternion& s, const quaternion& c, const quaternion& d, float t);
};

const vecd3 forward(1, 0, 0);
const vecd3 backword(-1, 0, 0);
const vecd3 right(0, 1, 0);
const vecd3 left(0, -1, 0);
const vecd3 up(0, 0, 1);
const vecd3 down(0, 0, -1);

const quaternion qid(1, 0, 0, 0);

#include "vec.cpp"

#endif
