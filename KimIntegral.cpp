#define _USE_MATH_DEFINES
#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>
#include <valarray>
#include <vector>
#include <functional>
using namespace std;

template<typename T>
double secant(double x1, double x2, double tol, T f)
{
    double xm, x0, c;
    if (f(x1) * f(x2) < 0) {
        do {
            x0 = (x1 * f(x2) - x2 * f(x1)) / (f(x2) - f(x1));
            c = f(x1) * f(x0);
            x1 = x2;
            x2 = x0;
            if (c == 0.0) { break; }
            xm = (x1 * f(x2) - x2 * f(x1)) / (f(x2) - f(x1));
        } while (fabs(xm - x0) >= tol);
        //cout << x0 << "\n";
        return x0;
    }
    else { return 0; }
}
template<typename T>
double bisect(double a, double b, double tol, int n, T f)
{
    int i = 0;
    double c = a;
    while ((b - a) >= tol) {
        c = (a + b) / 2;
        i++;
        if (f(c) == 0.0 || i >= n) { break; }
        else if (f(c) * f(a) < 0) { b = c; }
        else { a = c; }
    }
    //cout << c << "\n";
    return c;
}
double N(double x)
{
    return 0.5 * erfc(-x * M_SQRT1_2);
}
void print_varr(const valarray<double>& varr)
{
    cout << "[";
    for (size_t i = 0; i < varr.size(); i++) { cout << varr[i] << ", "; }
    cout << "]\n";
}
void print_vec(const vector<double>& vec)
{
    cout << "[";
    for (size_t i = 0; i < vec.size(); i++) { cout << vec[i] << ", "; }
    cout << "]";
}
valarray<double> linspace(double start, double stop, int size)
{
    valarray<double> v(size + 1);
    for (int i = 0; i < (size + 1); i++) v[i] = start + (i * (stop - start) / size);
    return v;
}
valarray<double> arange(double start, double step, double stop)
{
    int size = (stop - start) / step;
    valarray<double> v(size + 1);
    for (int i = 0; i < (size + 1); i++) v[i] = start + (step * i);
    return v;
}
valarray<double> append(valarray<double>& arr, double val)
{
    vector<double> vec;
    vec.assign(begin(arr), end(arr));
    vec.insert(vec.end(), val);
    arr = valarray<double>(vec.data(), vec.size());
    return arr;
}
//---------------------------------------------------------------------------------------------------------------------------------
class KimIntegral
{
public:
    KimIntegral() = default;
    KimIntegral(int, int, double, double, double, double, double, double, double, double, double);
    double* fit_boundary(int, bool);
    double* get_value(double*);
private:
    double H(double);
    double d1(double, double, double);
    double d2(double, double, double);
    double v(double, double, double);
    double f(double, double, double);
    double trapz(double, valarray<double>, bool);
    double residual_boundary(double, int, bool);
    double S, K, T, s, r, q, H0, H1, Kh, dt;
    const int M, R;
    valarray<double> B;

};
KimIntegral::KimIntegral(int R, int M, double S, double K, double T, double s, double r, double q, double H0, double H1, double Kh) :
    R(R), M(M), S(S), K(K), T(T), s(s), r(r), q(q), H0(H0), H1(H1), Kh(Kh)
{
    double Bm;
    if (q > 0. && r > 0.)
    {
        if (R == 1) { Bm = K * fmax(1., r / q); }
        else { Bm = K * fmin(1., r / q); }
    }
    else
    {
        Bm = K;
    }
    valarray<double> B0 = { Bm };
    B = B0;
    dt = T / M;
}
double KimIntegral::d1(double x, double y, double t)
{
    return (log(x / y) + (r - q + s * s * 0.5) * t) / (s * pow(t, 0.5));
}
double KimIntegral::d2(double x, double y, double t)
{
    return (log(x / y) + (r - q - s * s * 0.5) * t) / (s * pow(t, 0.5));
}
double KimIntegral::H(double x)
{
    double h;
    if (R == 1)
    {
        h = (H0 / 2) + (H1 / 2) * fmax(x - Kh, 0);
    }
    else
    {
        h = (H0 / 2) + (H1 / 2) * fmax(Kh - x, 0);
    }
    return h;
}
double KimIntegral::v(double x, double y, double t)
{
    if (R == 1)
    {
        return x * exp(-q * t) * N(d1(x, y, t)) - y * exp(-r * t) * N(d2(x, y, t));
    }
    else
    {
        return y * exp(-r * t) * N(-d2(x, y, t)) - x * exp(-q * t) * N(-d1(x, y, t));
    }
}
double KimIntegral::f(double x, double y, double t)
{
    if (R == 1)
    {
        if (t != 0.) { return q * x * exp(-q * t) * N(d1(x, y, t)) - r * K * exp(-r * t) * N(d2(x, y, t)); }
        else if (x < y) { return q * x - r * K; }
        else if (x > y) { return 0; }
        else { return 0.5 * (q * x - r * K); }
    }
    else
    {
        if (t != 0.) { return r * K * exp(-r * t) * N(-d2(x, y, t)) - q * x * exp(-q * t) * N(-d1(x, y, t)); }
        else if (x < y) { return r * K - q * x; }
        else if (x > y) { return 0; }
        else { return 0.5 * (r * K - q * x); }
    }
}
double KimIntegral::trapz(double x, valarray<double> y, bool append_x)
{
    valarray<double> t;
    double I = 0;
    if (append_x == true)
    {
        t = dt * arange(y.size(), -1, 0);
        for (size_t i = 1; i < y.size(); i++)
        {
            I += dt * f(x, y[i], t[i]);
        }
        I += dt / 2 * (f(x, y[0], t[0]) + f(x, x, 0));
    }
    else
    {
        t = dt * arange(y.size() - 1, -1, 0);
        for (size_t i = 1; i < y.size() - 1; i++)
        {
            I += dt * f(x, y[i], t[i]);
        }
        I += dt / 2 * (f(x, y[0], t[0]) + f(x, y[(y.size() - 1)], 0));
    }
    return I;
}
double KimIntegral::residual_boundary(double Bi, int i, bool spread)
{
    double eep = trapz(Bi, B, true);
    double eur = v(Bi, K, dt * i);
    double hsp, epsilon;
    if (spread == true)
    {
        hsp = H(Bi);
    }
    else
    {
        hsp = 0;
    }
    if (R == 1)
    {
        epsilon = K - Bi;
    }
    else
    {
        epsilon = Bi - K;
    }
    return epsilon + eur + eep - hsp;
}
double* KimIntegral::fit_boundary(int n, bool spread)
{
    for (int i = 1; i <= M; i++)
    {
        auto res = bind(&KimIntegral::residual_boundary, this, std::placeholders::_1, i, spread);
        double a = 0, b = 0;
        if (R == 1)
        {
            a = B[0];
            b = B[i - 1] * 2;
        }
        else
        {
            a = 0;
            b = B[0];
        }
        //double Bi = secant(a, b, 1e-9, res);
        double Bi = bisect(a, b, 1e-9, n, res);
        B = append(B, Bi);
    }
    double* Bx = new double[B.size()];
    copy(begin(B), end(B), Bx);
    return Bx;
}
double* KimIntegral::get_value(double* Bx)
{
    valarray<double> B(Bx, M + 1);
    double eur = v(S, K, T);
    double eep = trapz(S, B, false);
    double ame[2] = { eur, eep };
    return ame;
}

extern "C"
{
    KimIntegral* KI(int R, int M, double S, double K, double T, double s, double r, double q, double H0, double H1, double Kh)
    {
        return new KimIntegral(R, M, S, K, T, s, r, q, H0, H1, Kh);
    }
    double* fit_boundary(KimIntegral* ki, int n, bool spread)
    {
        return ki->fit_boundary(n, spread);
    }
    double* get_value(KimIntegral* ki, double* Bx)
    {
        return ki->get_value(Bx);
    }
}
