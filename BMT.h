#ifndef _BMT_
#define _BMT_

#include <cmath>
#include <limits>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <omp.h>
#include <immintrin.h>

enum class MatDataMajor : uint8_t { NaN = 0, Row = 1, Col = 2 }; // 描述矩阵形式的枚举类，“NaN”表示“未初始化”、“Row”表示“行优先存储”、“Col”表示“列优先存储”
enum class MatFileFormat : uint8_t { UNDEFINED = 0, BMT = 1};    // 描述外部导入矩阵数据的存储形式的枚举类，“BMT”是我自定义的一种采用行优先存储的二进制文件格式

// BMT格式的二进制矩阵数据文件包含了一个16位的文件头
struct BMTHeader
{
	uint32_t Off; // 到数据起始位置的位偏移(固定取值为16)
	uint32_t Typ; // 描述矩阵元素的类型，比如4-单精度浮点型(每个元素4个字节)、5-双精度浮点型(每个元素8个字节)[参照ENVI标准：https://www.nv5geospatialsoftware.com/docs/ENVIHeaderFiles.html]
	uint32_t Row; // 矩阵的行数
	uint32_t Col; // 矩阵的列数
};

// 自定义的简单矩阵类(只要用作管理矩阵数据的标准化容器)
class matrix
{
private:
	size_t Row;              // 矩阵的行数
	size_t Col;              // 矩阵的列数
	MatDataMajor Maj;        // 矩阵的形式(参见MatDataMajor枚举类的定义)
	std::vector<double> Mat; // 采用一维向量来存储矩阵元素，相较二维实现能更方便地进行内存复用(比如二维非对称阵的转置必须重新开辟内存)

	bool resize(size_t row_new, size_t col_new, MatDataMajor maj_new) // 调整矩阵的大小
	{
		if (row_new == 0 || col_new == 0)
		{
			fprintf(stderr, "矩阵的初始化大小不能为0！\n");
			return false;
		}
		if (maj_new == MatDataMajor::NaN)
		{
			fprintf(stderr, "不能使用未定义状态“NaN”来初始化矩阵！\n");
			return false;
		}

		Row = row_new;
		Col = col_new;
		Maj = maj_new;
		Mat.resize(Row * Col);
		if (Mat.capacity() > Mat.size() + 128) Mat.shrink_to_fit(); // 当vector容器容量较大于实际大小时，就释放掉多余的内存
		return true;
	}

public:
	size_t row() const       // 返回矩阵行数
	{
		return Row;
	}

	size_t col() const       // 返回矩阵列数
	{
		return Col;
	}

	MatDataMajor maj() const // 返回矩阵形式
	{
		return Maj;
	}

	void size(size_t& row_out, size_t& col_out) const // 通过引用同时取出矩阵的行数和列数
	{
		row_out = Row;
		col_out = Col;
	}

	void clear(bool ifShrinkMemory) // 清空矩阵、通过“ifShrinkMemory”开关来决定是否将vector容器内存也释放掉
	{
		Row = 0;
		Col = 0;
		Maj = MatDataMajor::NaN;
		if (!Mat.empty())
		{
			Mat.clear();
			if (ifShrinkMemory) Mat.shrink_to_fit();
		}
	}





	void LeastSquaresEstimation_Serial(const matrix& B, const matrix& l) // 最小二乘估计的粗糙串行实现
	{
		size_t obsv, para;  // “obsv”代表观测数量、“para”代表参数数量
		B.size(obsv, para); // 使用matrix类的size()函数取出设计矩阵B的维度、即观测数量和参数数量
		if (l.row() != obsv || l.col() != 1)
		{
			fprintf(stderr, "设计矩阵(%zu行%zu列)与观测向量(%zu行%zu列)的大小不匹配！\n", obsv, para, l.row(), l.col());
			exit(EXIT_FAILURE);
		}

		if (Maj != MatDataMajor::NaN) clear(true);
		resize(para, 1, MatDataMajor::Col); // 调整调用当前函数的matrix类对象的内存大小，用来存储法方程右端向量w和估计结果x(维度均为para行1列)



		// 第一步
		// 计算法方程系数矩阵Nbb = B' * B，此处串行实现采用的是同时适用于行优先存储和列优先存储的“()运算符重载”来访问矩阵元素
		matrix Nbb(para, para, MatDataMajor::Row, 0.0);
#if defined(_CoutTiming_)
		std::cout << "【粗糙串行实现】计时开始……\n";
		double T_stt = omp_get_wtime();
#endif
		for (size_t i = 0; i < para; i++)
		{
			for (size_t j = i; j < para; j++) // 利用Nbb的对称性、只需计算上三角部分
			{
				for (size_t k = 0; k < obsv; k++) Nbb(i, j) += B(k, i) * B(k, j); // 采用直接地址访问而非用临时局部变量取出元素进行运算、再将运算结果赋值回去，将大大增加计算开销
				if (i != j) Nbb(j, i) = Nbb(i, j); // 下三角部分直接复制上三角的计算结果
			}
		}
#if defined(_CoutTiming_)
		double T_end = omp_get_wtime();
		double T_sub = T_end - T_stt;
		double T_sum = T_sub;
		std::cout << "【粗糙串行实现】生成法方程系数矩阵耗时" << T_sub << "秒\n";
#endif



		// 第二步
		// 计算法方程右端向量w = B' * l、计算结果存储在调用当前函数的matrix类对象中
#if defined(_CoutTiming_)
		T_stt = omp_get_wtime();
#endif
		for (size_t n = 0; n < para; n++)
		{
			(*this)(n, 0) = 0.0;
			for (size_t m = 0; m < obsv; m++) (*this)(n, 0) += B(m, n) * l(m, 0); // 跟法方程系数矩阵的计算存在一样的问题，反复通过地址访问读写内存使程序性能低下
		}
#if defined(_CoutTiming_)
		T_end = omp_get_wtime();
		T_sub = T_end - T_stt;
		T_sum += T_sub;
		std::cout << "【粗糙串行实现】生成法方程右端向量耗时" << T_sub << "秒\n";
#endif



		// 第三步
		// 对法方程系数矩阵Nbb进行改进的Cholesky分解，得到Nbb = L * D * L'，其中L是对角元均为1的下三角矩阵、D是对角元均大于0的对角矩阵
#if defined(_CoutTiming_)
		T_stt = omp_get_wtime();
#endif
		for (size_t q = 0; q < para; q++) // 逐列进行LDL'分解
		{
			for (size_t r = 0; r < q; r++) Nbb(q, q) -= Nbb(r, q) * Nbb(q, r) * Nbb(r, r); // 计算对角矩阵D的第q个元素、结果存储到Nbb的对角线上

			for (size_t p = q + 1; p < para; p++) // 计算下三角矩阵L的第q列元素
			{
				for (size_t s = 0; s < q; s++) Nbb(p, q) -= Nbb(p, s) * Nbb(s, s) * Nbb(s, q);
				Nbb(p, q) /= Nbb(q, q);
				Nbb(q, p) = Nbb(p, q); // Nbb的下三角部分记录L、上三角部分记录L'，由对称性可得上三角部分取值
			}
		}
#if defined(_CoutTiming_)
		T_end = omp_get_wtime();
		T_sub = T_end - T_stt;
		T_sum += T_sub;
		std::cout << "【粗糙串行实现】法方程系数矩阵的改进Cholesky分解耗时" << T_sub << "秒\n";
#endif



		// 第四步
		//将法方程系数矩阵Nbb分解为下三角矩阵L、对角矩阵D和上三角矩阵L'之后，采用前代法、行变换(除以对角元)和回代法完成法方程的求解
#if defined(_CoutTiming_)
		T_stt = omp_get_wtime();
#endif
		for (size_t u = 0; u < para - 1; u++) // 使用前代法解系数矩阵为下三角矩阵的线性方程组
		{
			for (size_t v = u + 1; v < para; v++) (*this)(v, 0) -= Nbb(v, u) * (*this)(u, 0);
		}

		for (size_t t = 0; t < para; t++) (*this)(t, 0) /= Nbb(t, t); // 左除对角矩阵即每一行乘以对应对角元素的倒数(等号右侧为向量、因此每一行只有一个元素需要操作)

		for (size_t y = para - 1; y > 0; y--) // 使用回代法解系数矩阵为上三角矩阵的线性方程组
		{
			for (size_t x = 0; x < y; x++) (*this)(x, 0) -= Nbb(x, y) * (*this)(y, 0);
		}
#if defined(_CoutTiming_)
		T_end = omp_get_wtime();
		T_sub = T_end - T_stt;
		T_sum += T_sub;
		std::cout << "【粗糙串行实现】基于改进Cholesky分解的线性方程组求解耗时" << T_sub << "秒\n";
		std::cout << "【粗糙串行实现】累计共耗时" << T_sum << "秒\n\n";
#endif
	}





	void LeastSquaresEstimation_Parallel(const matrix& B, const matrix& l, bool on_off) // 最小二乘估计的最佳串行实现及其OpenMP直接并行化、其中“on_off”就是OpenMP并行化的开关
	{
		size_t obsv, para;
		B.size(obsv, para);
		if (B.maj() != MatDataMajor::Col) // Nbb = B' * B的第i行、第j列元素即矩阵B的第i列与第j列的内积，当采用列优先存储时矩阵B的每一列都内存连续、缓存命中率大大增加
		{
			fprintf(stderr, "参与并发最小二乘的设计矩阵必须采用列优先存储！\n");
			exit(EXIT_FAILURE);
		}
		if (l.row() != obsv || l.col() != 1)
		{
			fprintf(stderr, "设计矩阵(%zu行%zu列)与观测向量(%zu行%zu列)的大小不匹配！\n", obsv, para, l.row(), l.col());
			exit(EXIT_FAILURE);
		}

		if (Maj != MatDataMajor::NaN) clear(true);
		resize(para, 1, MatDataMajor::Col); // 调整调用该函数的matrix类对象的内存大小，用来存储法方程右端向量w和估计结果x(维度均为para行1列)
		


		// 第一步
		// 计算法方程系数矩阵Nbb = B' * B
		matrix Nbb(para, para, MatDataMajor::Row, 0.0);
#if defined(_CoutTiming_)
		if (on_off) std::cout << "【OpenMP并行实现】计时开始……\n";
		else std::cout << "【最佳串行实现】计时开始……\n";
		double T_stt = omp_get_wtime();
#endif
		const int Wid = static_cast<int>(para);
		const int Hgt = static_cast<int>(obsv);
#pragma omp parallel for schedule(dynamic, 4) num_threads(16) if(on_off)    // 每个线程负责Nbb的4行元素的计算，由于Nbb的第i行需计算的元素数量为para - i、各线程负载必然不同，故采用dynamic动态调度
		for (int i = 0; i < Wid; i++)
		{
			const double* pCol_i = B.Mat.data() + i * Hgt;                  // 设计矩阵B第i列的首指针
			for (int j = i; j < Wid; j++)
			{
				const double* pCol_j = B.Mat.data() + j * Hgt;              // 设计矩阵B第j列的首指针

				double Inp = 0.0;                                           // 使用局部临时变量存储内积结果
				for (int k = 0; k < Hgt; k++) Inp += pCol_i[k] * pCol_j[k]; // 进行设计矩阵B的第i列和第j列的内积计算
				Nbb.Mat[i * Wid + j] = Inp;                                 // 将计算结果存储到矩阵Nbb内
				if (i != j) Nbb.Mat[j * Wid + i] = Inp;                     // 更新对称侧元素
			}
		}
#if defined(_CoutTiming_)
		double T_end = omp_get_wtime();
		double T_sub = T_end - T_stt;
		double T_sum = T_sub;
		if (on_off) std::cout << "【OpenMP并行实现】生成法方程系数矩阵耗时" << T_sub << "秒\n";
		else std::cout << "【最佳串行实现】生成法方程系数矩阵耗时" << T_sub << "秒\n";
#endif



		// 第二步
		// 计算法方程右端向量w = B' * l
#if defined(_CoutTiming_)
		T_stt = omp_get_wtime();
#endif
		const double* pCol_l = l.Mat.data();                            // 观测向量l的首指针
#pragma omp parallel for schedule(static) num_threads(16) if(on_off)
		for (int p = 0; p < Wid; p++)
		{
			const double* pCol_b = B.Mat.data() + p * Hgt;              // 设计矩阵B第p列的首指针

			double Inp = 0.0;                                           // 使用局部临时变量存储内积结果
			for (int q = 0; q < Hgt; q++) Inp += pCol_b[q] * pCol_l[q]; // 进行设计矩阵B的第p列和观测向量l的内积计算
			(*this).Mat[p] = Inp;                                       // 将计算结果存储到调用该函数的矩阵内
		}
#if defined(_CoutTiming_)
		T_end = omp_get_wtime();
		T_sub = T_end - T_stt;
		T_sum += T_sub;
		if (on_off) std::cout << "【OpenMP并行实现】生成法方程右端向量耗时" << T_sub << "秒\n";
		else std::cout << "【最佳串行实现】生成法方程右端向量耗时" << T_sub << "秒\n";
#endif



		// 第三步
		// 对法方程系数矩阵Nbb进行LDL'分解
#if defined(_CoutTiming_)
		T_stt = omp_get_wtime();
#endif
		std::vector<double> D(para, 0.0); // 使用一个额外的临时容器来连续存储对角矩阵D的对角元素，以便在后续使用时提高缓存命中率
		const double* pVec_d = D.data();  // 对称矩阵D的对角元素向量的首指针
		for (int r = 0; r < Wid; r++)
		{
			// 首先计算对角矩阵D的第r个对角元
			double subt = 0.0;
			const double* pRow_r = Nbb.Mat.data() + r * Wid;                       // 矩阵Nbb第r行的首指针，此处实际访问的是存储在矩阵Nbb中的前几次分解结果
			for (int s = 0; s < r; s++) subt += pRow_r[s] * pRow_r[s] * pVec_d[s];
			double diag = Nbb.Mat[r * Wid + r] - subt;                             // 使用局部临时变量存储第r个对角元计算结果、以便接下来计算下三角矩阵L的第r列元素的使用
			Nbb.Mat[r * Wid + r] = 1.0;                                            // 下三角矩阵L的对角元素为1(该步骤实际多余、只为体现下三角矩阵L的性质)
			D[r] = diag;                                                           // 将对角元的计算结果给保存到容器D内

			if (diag < 0.0 || std::fabs(diag) < 1e-12)
			{
				fprintf(stderr, "法方程系数矩阵奇异，无法完成最小二乘解算！");
				exit(EXIT_FAILURE);
			}

			// 其次计算下三角矩阵L的第r列元素，该步骤只需前para - 1列(索引为0 ~ para - 2、para = Wid)进行，因为最后一列只有对角元、对角线以下没有元素
			if (r < Wid - 1)
			{
#pragma omp parallel for schedule(static) num_threads(8) if (r < Wid - 32 && on_off)  // 当r >= Wid - 32时，第r列的下三角元素小于32个、各线程分配到的任务少于4个，此时多线程调度开销将大于并行化收益
				for (int u = r + 1; u < Wid; u++)
				{
					double subt = 0.0;
					const double* pRow_u = Nbb.Mat.data() + u * Wid;
					for (int v = 0; v < r; v++) subt += pRow_r[v] * pRow_u[v] * D[v]; // 累加量只跟r有关、即第r列各下三角元素的计算量一致，因此采用static静态调度即可保证负载均衡

					double elem = Nbb.Mat[r * Wid + u];
					Nbb.Mat[r * Wid + u] = (elem - subt) / diag;
					Nbb.Mat[u * Wid + r] = (elem - subt) / diag; // 同时更新对称部分，以便于后续前代法和回代法解线性方程组的使用
				}
			}
		}
#if defined(_CoutTiming_)
		T_end = omp_get_wtime();
		T_sub = T_end - T_stt;
		T_sum += T_sub;
		if (on_off) std::cout << "【OpenMP并行实现】法方程系数矩阵的改进Cholesky分解耗时" << T_sub << "秒\n";
		else std::cout << "【最佳串行实现】法方程系数矩阵的改进Cholesky分解耗时" << T_sub << "秒\n";
#endif



		// 第四步
		// 线性方程组Nbb * x = w在完成矩阵分解之后被转化为 (L * D * L') * x = w，其中L为对角元均为1的下三角矩阵、D为对角元均大于0的对角矩阵，只需依次采用前代法、行变换(每一行除以对角阵D中位于相同行的对角元素)和回代法即可快速完成线性方程组的求解
#if defined(_CoutTiming_)
		T_stt = omp_get_wtime();
#endif
		// 前代法：此时Nbb的上三角部分存储了L'、下三角部分存储了L，前代法使用到的是下三角的L，但是Nbb采用行优先存储、而前代法需要沿列方向访问，对此可通过沿行方向访问上三角的L'来解决、从而提升缓存命中率(因为沿行方向内存连续)
		double* pVec_w = (*this).Mat.data();                                   // 法方程右端向量w的首指针
		for (int m = 0; m < Wid - 1; m++)
		{
			const double val_Wm = (*this).Mat[m];                              // 取出Wm
			const double* pUpp_m = Nbb.Mat.data() + m * Wid;                   // 矩阵Nbb第m行的首指针、此处相当于下三角矩阵L第m列的首指针
			for (int n = m + 1; n < Wid; n++) pVec_w[n] -= val_Wm * pUpp_m[n]; // Wn -= Wm * Lnm
		}

		// 行变换
		for (int h = 0; h < Wid; h++) pVec_w[h] /= pVec_d[h];                  // Wh /= Dhh

		// 回代法：回代法使用到的是上三角的L'并沿列方向进行访问，跟前代法的实现类似、通过沿行方向读取Nbb下三角的L来保证内存连续和缓存命中率
		for (int f = Wid - 1; f > 0; f--)
		{
			const double val_Wf = (*this).Mat[f];                              // 取出Wf
			const double* pLow_f = Nbb.Mat.data() + f * Wid;                   // 矩阵Nbb第f行的首指针、此处相当于上三角矩阵L'第f列的首指针
			for (int g = 0; g < f; g++) pVec_w[g] -= val_Wf * pLow_f[g];       // Wg -= Wf * L'gf
		}
#if defined(_CoutTiming_)
		T_end = omp_get_wtime();
		T_sub = T_end - T_stt;
		T_sum += T_sub;
		if (on_off)
		{
			std::cout << "【OpenMP并行实现】基于改进Cholesky分解的线性方程组求解耗时" << T_sub << "秒\n";
			std::cout << "【OpenMP并行实现】累计共耗时" << T_sum << "秒\n\n";
		}
		else
		{
			std::cout << "【最佳串行实现】基于改进Cholesky分解的线性方程组求解耗时" << T_sub << "秒\n";
			std::cout << "【最佳串行实现】累计共耗时" << T_sum << "秒\n\n";
		}
#endif
	}





	void LeastSquaresEstimation_Parallel_AVX(const matrix& B, const matrix& l) // 在OpenMP直接并行化的基础上，通过引入向量化来继续提高计算效率，接下来将仅针对向量化实现进行介绍、各模块的实现详见“LeastSquaresEstimation_Parallel()”的相关注释和实习报告
	{
		size_t obsv, para;
		B.size(obsv, para);
		if (B.maj() != MatDataMajor::Col)
		{
			fprintf(stderr, "参与并发最小二乘的设计矩阵必须采用列优先存储！\n");
			exit(EXIT_FAILURE);
		}
		if (l.row() != obsv || l.col() != 1)
		{
			fprintf(stderr, "设计矩阵(%zu行%zu列)与观测向量(%zu行%zu列)的大小不匹配！\n", obsv, para, l.row(), l.col());
			exit(EXIT_FAILURE);
		}

		if (Maj != MatDataMajor::NaN) clear(true);
		resize(para, 1, MatDataMajor::Col);



		// 第一、二步
		// 计算法方程的系数矩阵Nbb = B' * B和右端向量w = B' * l
		matrix Nbb(para, para, MatDataMajor::Row, 0.0);
#if defined(_CoutTiming_)
		std::cout << "【OpenMP(AVX2)并行实现】计时开始……\n";
		double T_stt = omp_get_wtime();
#endif
		const int Wid = static_cast<int>(para);
		const int Hgt = static_cast<int>(obsv);
		const int Tol = static_cast<int>(obsv) - 16;
		const double* pCol_l = l.Mat.data();

#pragma omp parallel num_threads(16)
		{
			__m256d accu_1, accu_2, accu_3, accu_4; // 声明每个线程私有的累加器，每个寄存器可同时对4个双精度浮点型数据进行计算、而四个寄存器则对应16个双精度浮点型数据

			// 计算法方程的系数矩阵Nbb
#pragma omp for schedule(dynamic, 4)
			for (int i = 0; i < Wid; i++)
			{
				const double* pCol_i = B.Mat.data() + i * Hgt;
				for (int j = i; j < Wid; j++)
				{
					const double* pCol_j = B.Mat.data() + j * Hgt;

					// 累加器向量初始化
					accu_1 = _mm256_setzero_pd();
					accu_2 = _mm256_setzero_pd();
					accu_3 = _mm256_setzero_pd();
					accu_4 = _mm256_setzero_pd();

					int k = 0;
					for (; k <= Tol; k += 16) // 每次取16个双精度浮点型数据
					{
						// 提前将下一次要访问的数据加载到L1缓存、以减少因缓存不命中造成的延迟
						_mm_prefetch(reinterpret_cast<const char*>(pCol_i + k + 16), _MM_HINT_T0);
						_mm_prefetch(reinterpret_cast<const char*>(pCol_j + k + 16), _MM_HINT_T0);

						// 每个向量取出4个双精度浮点型数据以进行局部内积计算
						__m256d vec_i1 = _mm256_loadu_pd(pCol_i + k);
						__m256d vec_i2 = _mm256_loadu_pd(pCol_i + k + 4);
						__m256d vec_i3 = _mm256_loadu_pd(pCol_i + k + 8);
						__m256d vec_i4 = _mm256_loadu_pd(pCol_i + k + 12);

						__m256d vec_j1 = _mm256_loadu_pd(pCol_j + k);
						__m256d vec_j2 = _mm256_loadu_pd(pCol_j + k + 4);
						__m256d vec_j3 = _mm256_loadu_pd(pCol_j + k + 8);
						__m256d vec_j4 = _mm256_loadu_pd(pCol_j + k + 12);

						// 通过“融合乘加”操作将局部(4个)内积结果更新到累加器中
						accu_1 = _mm256_fmadd_pd(vec_i1, vec_j1, accu_1);
						accu_2 = _mm256_fmadd_pd(vec_i2, vec_j2, accu_2);
						accu_3 = _mm256_fmadd_pd(vec_i3, vec_j3, accu_3);
						accu_4 = _mm256_fmadd_pd(vec_i4, vec_j4, accu_4);
					}
					// 完成能被16整除部分的内积操作之后、聚合各累加器的计算结果
					accu_1 = _mm256_add_pd(accu_1, accu_2);
					accu_3 = _mm256_add_pd(accu_3, accu_4);
					accu_1 = _mm256_add_pd(accu_1, accu_3);

					__m128d vec_2l = _mm256_castpd256_pd128(accu_1);           // 提取低128位(2个双精度浮点型数据)，通过直接类型转换实现、相较“_mm256_extractf128_pd(accu_1, 0)”在理论上效率更高
					                                                           // accu_1: [a, b, c, d];   vec_2l: [a, b]
					__m128d vec_2h = _mm256_extractf128_pd(accu_1, 1);         // 提取高128位(2个双精度浮点型数据)
					                                                           // accu_1: [a, b, c, d];   vec_2h: [c, d]
					vec_2l = _mm_add_pd(vec_2l, vec_2h);                       // 将两个128位向量相加
					                                                           // vec_2l: [a + c, b + d]; vec_2h: [c, d]
					vec_2h = _mm_unpackhi_pd(vec_2l, vec_2l);                  // 从两个128位向量提取各自的高64位组成新的128位向量，此处vec_2h的两个64位均存储vec_2l的高64位
					                                                           // vec_2l: [a + c, b + d]; vec_2h: [b + d, b + d]
					double result = _mm_cvtsd_f64(_mm_add_sd(vec_2l, vec_2h)); // vec_2l和vec_2h相加结果的低64位等于vec_2l的高64位和低64位的和，“_mm_cvtsd_f64”将128位向量的低64位转化为双精度浮点数
					                                                           // vec_2l + vec_2h: [a + b + c + d, b + b + d + d] -> return: [a + b + c + d]

					for (; k < Hgt; k++) result += pCol_i[k] * pCol_j[k];      // 完成剩余部分的内积操作
					Nbb.Mat[i * Wid + j] = result;                             // 将结果保存到矩阵Nbb内
					if (i != j) Nbb.Mat[j * Wid + i] = result;                 // 同时保存对称部分
				}
			}

			// 计算法方程的右端向量w
#pragma omp for schedule(static)
			for (int p = 0; p < Wid; p++)
			{
				const double* pCol_b = B.Mat.data() + p * Hgt;

				// 累加器向量初始化
				accu_1 = _mm256_setzero_pd();
				accu_2 = _mm256_setzero_pd();
				accu_3 = _mm256_setzero_pd();
				accu_4 = _mm256_setzero_pd();

				int q = 0;
				for (; q <= Tol; q += 16) // 每次取16个双精度浮点型数据
				{
					// 提前将下一次要访问的数据加载到L1缓存、以减少因缓存不命中造成的延迟
					_mm_prefetch(reinterpret_cast<const char*>(pCol_b + q + 16), _MM_HINT_T0);
					_mm_prefetch(reinterpret_cast<const char*>(pCol_l + q + 16), _MM_HINT_T0);

					// 每个向量取出4个双精度浮点型数据以进行局部内积计算
					__m256d vec_b1 = _mm256_loadu_pd(pCol_b + q);
					__m256d vec_b2 = _mm256_loadu_pd(pCol_b + q + 4);
					__m256d vec_b3 = _mm256_loadu_pd(pCol_b + q + 8);
					__m256d vec_b4 = _mm256_loadu_pd(pCol_b + q + 12);

					__m256d vec_l1 = _mm256_loadu_pd(pCol_l + q);
					__m256d vec_l2 = _mm256_loadu_pd(pCol_l + q + 4);
					__m256d vec_l3 = _mm256_loadu_pd(pCol_l + q + 8);
					__m256d vec_l4 = _mm256_loadu_pd(pCol_l + q + 12);

					// 通过“融合乘加”操作将局部(4个)内积结果更新到累加器中
					accu_1 = _mm256_fmadd_pd(vec_b1, vec_l1, accu_1);
					accu_2 = _mm256_fmadd_pd(vec_b2, vec_l2, accu_2);
					accu_3 = _mm256_fmadd_pd(vec_b3, vec_l3, accu_3);
					accu_4 = _mm256_fmadd_pd(vec_b4, vec_l4, accu_4);
				}
				// 完成能被16整除部分的内积操作之后、聚合各累加器的计算结果
				accu_1 = _mm256_add_pd(accu_1, accu_2);
				accu_3 = _mm256_add_pd(accu_3, accu_4);
				accu_1 = _mm256_add_pd(accu_1, accu_3);

				__m128d vec_2l = _mm256_castpd256_pd128(accu_1);           // 提取低128位(2个双精度浮点型数据)，通过直接类型转换实现、相较“_mm256_extractf128_pd(accu_1, 0)”在理论上效率更高
				                                                           // accu_1: [a, b, c, d];   vec_2l: [a, b]
				__m128d vec_2h = _mm256_extractf128_pd(accu_1, 1);         // 提取高128位(2个双精度浮点型数据)
				                                                           // accu_1: [a, b, c, d];   vec_2h: [c, d]
				vec_2l = _mm_add_pd(vec_2l, vec_2h);                       // 将两个128位向量相加
				                                                           // vec_2l: [a + c, b + d]; vec_2h: [c, d]
				vec_2h = _mm_unpackhi_pd(vec_2l, vec_2l);                  // 从两个128位向量提取各自的高64位组成新的128位向量，此处vec_2h的两个64位均存储vec_2l的高64位
				                                                           // vec_2l: [a + c, b + d]; vec_2h: [b + d, b + d]
				double result = _mm_cvtsd_f64(_mm_add_sd(vec_2l, vec_2h)); // vec_2l和vec_2h相加结果的低64位等于vec_2l的高64位和低64位的和，“_mm_cvtsd_f64”将128位向量的低64位转化为双精度浮点数
				                                                           // vec_2l + vec_2h: [a + b + c + d, b + b + d + d] -> return: [a + b + c + d]

				for (; q < Hgt; q++) result += pCol_b[q] * pCol_l[q];      // 完成剩余部分的内积操作
				(*this).Mat[p] = result;                                   // 将结果保存到调用该函数的矩阵内
			}
		}
#if defined(_CoutTiming_)
		double T_end = omp_get_wtime();
		double T_sub = T_end - T_stt;
		double T_sum = T_sub;
		std::cout << "【OpenMP(AVX2)并行实现】生成法方程共耗时" << T_sub << "秒\n";
#endif



		// 第三步
		// 法方程的系数矩阵Nbb的改进Cholesky分解
#if defined(_CoutTiming_)
		T_stt = omp_get_wtime();
#endif
		std::vector<double> D(para, 0.0);
		const double* pVec_d = D.data();
		for (int r = 0; r < Wid; r++)
		{
			// 首先计算对角矩阵D的第r个对角元
			double diag = Nbb.Mat[r * Wid + r];
			const double* pRow_r = Nbb.Mat.data() + r * Wid;
			if (r < 4) // 前4列需进行累加计算的次数小于4、无需使用向量化
			{
				for (int s = 0; s < r; s++) diag -= Nbb.Mat[r * Wid + s] * Nbb.Mat[r * Wid + s] * D[s];
				D[r] = diag;
			}
			else
			{
				__m256d accu_d = _mm256_setzero_pd();               //声明并初始化累加器

				int s = 0;
				for (; s <= r - 4; s += 4)                          // 每次取4个双精度浮点型数据进行计算
				{
					__m256d vec_rs = _mm256_loadu_pd(pRow_r + s);   // 取出Lrs ~ Lr(s+3)
					__m256d vec_ds = _mm256_loadu_pd(pVec_d + s);   // 取出Dss ~ D(s+3)(s+3)
					__m256d mul_rd = _mm256_mul_pd(vec_rs, vec_rs); // 计算Lri * Lri * Dii、其中i = s ~ s+3
					mul_rd = _mm256_mul_pd(mul_rd, vec_ds);
					accu_d = _mm256_add_pd(accu_d, mul_rd);         // 累加计算结果
				}

				// 同上将256位向量中的4个双精度浮点数进行累加和传出
				__m128d vec_2l = _mm256_castpd256_pd128(accu_d);
				__m128d vec_2h = _mm256_extractf128_pd(accu_d, 1);
				vec_2l = _mm_add_pd(vec_2l, vec_2h);
				vec_2h = _mm_unpackhi_pd(vec_2l, vec_2l);
				diag -= _mm_cvtsd_f64(_mm_add_sd(vec_2l, vec_2h));

				// 完成剩余部分的计算
				for (; s < r; s++) diag -= Nbb.Mat[r * Wid + s] * Nbb.Mat[r * Wid + s] * D[s];
				D[r] = diag;
			}
			Nbb.Mat[r * Wid + r] = 1.0;

			if (diag < 0.0 || std::fabs(diag) < 1e-12)
			{
				fprintf(stderr, "法方程系数矩阵奇异，无法完成最小二乘解算！");
				exit(EXIT_FAILURE);
			}

			// 其次计算下三角矩阵L的第r列元素
			if (r < Wid - 1)
			{
				if (r < 16) // 前几列无需使用到向量化，简单并行优化即可
				{
#pragma omp parallel for schedule(static) num_threads(8)
					for (int u = r + 1; u < Wid; u++)
					{
						double elem = Nbb.Mat[r * Wid + u];
						for (int v = 0; v < r; v++) elem -= Nbb.Mat[r * Wid + v] * Nbb.Mat[u * Wid + v] * D[v];
						Nbb.Mat[r * Wid + u] = elem / diag;
						Nbb.Mat[u * Wid + r] = elem / diag;
					}
				}
				else
				{
#pragma omp parallel num_threads(8) if (r < Wid - 32)
					{
						__m256d accu_o; // 声明各线程私有的累加器

#pragma omp for schedule(static)
						for (int u = r + 1; u < Wid; u++)
						{
							accu_o = _mm256_setzero_pd(); // 初始化累加器

							double elem = Nbb.Mat[r * Wid + u];
							const double* pRow_u = Nbb.Mat.data() + u * Wid;

							int v = 0;
							for (; v <= r - 4; v += 4)
							{
								// 计算Luv * Lvr * Dvv并关于v进行累加(由于对称性有Luv = Lvu、Lvr = Lrv)
								__m256d vec_rv = _mm256_loadu_pd(pRow_r + v);
								__m256d vec_uv = _mm256_loadu_pd(pRow_u + v);
								__m256d vec_dv = _mm256_loadu_pd(pVec_d + v);
								__m256d mul_rd = _mm256_mul_pd(vec_rv, vec_uv);
								mul_rd = _mm256_mul_pd(mul_rd, vec_dv);
								accu_o = _mm256_add_pd(accu_o, mul_rd);
							}

							// 同上将256位向量中的4个双精度浮点数进行累加和传出
							__m128d vec_2l = _mm256_castpd256_pd128(accu_o);
							__m128d vec_2h = _mm256_extractf128_pd(accu_o, 1);
							vec_2l = _mm_add_pd(vec_2l, vec_2h);
							vec_2h = _mm_unpackhi_pd(vec_2l, vec_2l);
							elem -= _mm_cvtsd_f64(_mm_add_sd(vec_2l, vec_2h));

							// 完成剩余部分的计算
							for (; v < r; v++) elem -= Nbb.Mat[r * Wid + v] * Nbb.Mat[u * Wid + v] * D[v];
							Nbb.Mat[r * Wid + u] = elem / diag;
							Nbb.Mat[u * Wid + r] = elem / diag;
						}
					}
				}
			}
		}
#if defined(_CoutTiming_)
		T_end = omp_get_wtime();
		T_sub = T_end - T_stt;
		T_sum += T_sub;
		std::cout << "【OpenMP(AVX2)并行实现】法方程系数矩阵的改进Cholesky分解耗时" << T_sub << "秒\n";
#endif
		


		// 第四步
		// 求解线性方程组Nbb * x = (L * D * L') * x = w
#if defined(_CoutTiming_)
		T_stt = omp_get_wtime();
#endif
		// 前代法：如前所述是沿行方向访问上三角的L'来代替沿列方向访问下三角的L
		double* pVec_w = (*this).Mat.data();
		for (int m = 0; m < Wid - 1; m++)
		{
			const double val_Wm = (*this).Mat[m];
			const double* pUpp_m = Nbb.Mat.data() + m * Wid;
			if (m < Wid - 4)
			{
				__m256d vec_Wm = _mm256_set1_pd(val_Wm);

				int n = m + 1;
				for (; n <= Wid - 4; n += 4) // 批量进行Wn -= Wm * Lnm
				{
					__m256d vec_Ln = _mm256_loadu_pd(pUpp_m + n);
					__m256d vec_Wn = _mm256_loadu_pd(pVec_w + n);
					vec_Ln = _mm256_mul_pd(vec_Ln, vec_Wm);
					vec_Wn = _mm256_sub_pd(vec_Wn, vec_Ln);
					_mm256_storeu_pd(pVec_w + n, vec_Wn);
				}
				for (; n < Wid; n++) pVec_w[n] -= val_Wm * pUpp_m[n];
			}
			else // 最后四列剩余元素数量不足以进行向量化，直接普通串行实现
			{
				for (int n = m + 1; n < Wid; n++) pVec_w[n] -= val_Wm * pUpp_m[n];
			}
		}

		// 行变换：实现法方程右端向量w和对角矩阵D的对角元素的逐个相除
		int h = 0;
		for (; h <= Wid - 4; h += 4)
		{
			__m256d vec_Dh = _mm256_loadu_pd(pVec_d + h);
			__m256d vec_Wh = _mm256_loadu_pd(pVec_w + h);
			vec_Wh = _mm256_div_pd(vec_Wh, vec_Dh);
			_mm256_storeu_pd(pVec_w + h, vec_Wh);
		}
		for (; h < Wid; h++) pVec_w[h] /= pVec_d[h];

		// 回代法：如前所述是沿行方向访问下三角的L来代替沿列方向访问上三角的L'
		for (int f = Wid - 1; f > 0; f--)
		{
			const double val_Wf = (*this).Mat[f];
			const double* pLow_f = Nbb.Mat.data() + f * Wid;
			if (f > 3)
			{
				__m256d vec_Wf = _mm256_set1_pd(val_Wf);

				int g = 0;
				for (; g <= f - 4; g += 4) // 批量进行Wg -= Wf * L'gf
				{
					__m256d vec_Lg = _mm256_loadu_pd(pLow_f + g);
					__m256d vec_Wg = _mm256_loadu_pd(pVec_w + g);
					vec_Lg = _mm256_mul_pd(vec_Lg, vec_Wf);
					vec_Wg = _mm256_sub_pd(vec_Wg, vec_Lg);
					_mm256_storeu_pd(pVec_w + g, vec_Wg);
				}
				for (; g < f; g++) pVec_w[g] -= val_Wf * pLow_f[g];
			}
			else // 最后四列(位于上三角矩阵L'的左侧、回代法是从右向左进行的)剩余元素数量不足以进行向量化，直接普通串行实现
			{
				for (int g = 0; g < f; g++) pVec_w[g] -= val_Wf * pLow_f[g];
			}
		}
#if defined(_CoutTiming_)
		T_end = omp_get_wtime();
		T_sub = T_end - T_stt;
		T_sum += T_sub;
		std::cout << "【OpenMP(AVX2)并行实现】基于改进Cholesky分解的线性方程组求解耗时" << T_sub << "秒\n";
		std::cout << "【OpenMP(AVX2)并行实现】累计共耗时" << T_sum << "秒\n\n";
#endif
	}





	double& operator()(size_t i, size_t j)
	{
		if (Maj == MatDataMajor::NaN)
		{
			fprintf(stderr, "当前矩阵为空！");
			exit(EXIT_FAILURE);
		}

		if (i >= Row || j >= Col)
		{
			fprintf(stderr, "下标索引超出范围！");
			exit(EXIT_FAILURE);
		}

		if (Maj == MatDataMajor::Row)
		{
			return Mat[i * Col + j];
		}
		else if (Maj == MatDataMajor::Col)
		{
			return Mat[j * Row + i];
		}
		else
		{
			fprintf(stderr, "进入有关枚举类MatDataMajor的非法分支！");
			exit(EXIT_FAILURE);
		}
	}



	const double& operator()(size_t i, size_t j) const
	{
		if (Maj == MatDataMajor::NaN)
		{
			fprintf(stderr, "当前矩阵为空！");
			exit(EXIT_FAILURE);
		}

		if (i >= Row || j >= Col)
		{
			fprintf(stderr, "下标索引超出范围！");
			exit(EXIT_FAILURE);
		}

		if (Maj == MatDataMajor::Row)
		{
			return Mat[i * Col + j];
		}
		else if (Maj == MatDataMajor::Col)
		{
			return Mat[j * Row + i];
		}
		else
		{
			fprintf(stderr, "进入有关枚举类MatDataMajor的非法分支！");
			exit(EXIT_FAILURE);
		}
	}



	matrix() : Row(0), Col(0), Maj(MatDataMajor::NaN), Mat() {}



	matrix(size_t row_new, size_t col_new, MatDataMajor maj_new)
	{
		if (row_new == 0 || col_new == 0)
		{
			fprintf(stderr, "矩阵的初始化大小不能为0！\n");
			exit(EXIT_FAILURE);
		}
		if (maj_new == MatDataMajor::NaN)
		{
			fprintf(stderr, "不能使用未定义状态“NaN”来初始化矩阵！\n");
			exit(EXIT_FAILURE);
		}

		Row = row_new;
		Col = col_new;
		Maj = maj_new;
		Mat.resize(Row * Col);
	}



	matrix(size_t row_new, size_t col_new, MatDataMajor maj_new, double val_ini) // “val_ini”是初始化矩阵所用值
	{
		if (row_new == 0 || col_new == 0)
		{
			fprintf(stderr, "矩阵的初始化大小不能为0！\n");
			exit(EXIT_FAILURE);
		}
		if (maj_new == MatDataMajor::NaN)
		{
			fprintf(stderr, "不能使用未定义状态“NaN”来初始化矩阵！\n");
			exit(EXIT_FAILURE);
		}

		Row = row_new;
		Col = col_new;
		Maj = maj_new;
		Mat.resize(Row * Col, val_ini);
	}



	matrix(const std::string& FilePath, MatFileFormat FileFormat, MatDataMajor DataMajor) // 用于读取“BMT”二进制格式矩阵数据的构造函数
	{
		if (DataMajor == MatDataMajor::NaN)
		{
			fprintf(stderr, "不能使用未定义状态“NaN”来初始化矩阵！\n");
			exit(EXIT_FAILURE);
		}
		Maj = DataMajor;

		if (FileFormat == MatFileFormat::BMT)
		{
			std::ifstream MatData(FilePath, std::ios::binary);
			if (!MatData.is_open())
			{
				fprintf(stderr, "无法打开该矩阵数据文件：%s！\n", FilePath.c_str());
				exit(EXIT_FAILURE);
			}

			BMTHeader MatHead = {};
			MatData.read(reinterpret_cast<char*>(&MatHead), sizeof(MatHead));
			if (MatHead.Off != 16)
			{
				fprintf(stderr, "文件头长度与BMT格式不符！\n");
				exit(EXIT_FAILURE);
			}
			Row = static_cast<size_t>(MatHead.Row);
			Col = static_cast<size_t>(MatHead.Col);
			size_t Num = Row * Col;
			Mat.resize(Num);

			if (MatHead.Typ == 4) // 单精度浮点型
			{
				float* MatBuff = new float[Num];
				MatData.read(reinterpret_cast<char*>(MatBuff), Num * sizeof(float));
				if (DataMajor == MatDataMajor::Row)
				{
					int Tol = static_cast<int>(Num);
#pragma omp parallel for schedule(static, 64) num_threads(8) if(Num > 8192)
					for (int i = 0; i < Tol; i++) Mat[i] = static_cast<double>(MatBuff[i]);
				}
				else if (DataMajor == MatDataMajor::Col)
				{
					int Tol = static_cast<int>(Num);
#pragma omp parallel for schedule(static, 64) num_threads(8) if(Num > 8192)
					for (int j = 0; j < Tol; j++)
					{
						size_t m = static_cast<size_t>(j) / Col;
						size_t n = static_cast<size_t>(j) % Col;
						Mat[n * Row + m] = static_cast<double>(MatBuff[j]);
					}
				}
				else
				{
					fprintf(stderr, "进入有关枚举类MatDataMajor的非法分支！");
					exit(EXIT_FAILURE);
				}
				delete[] MatBuff;
			}
			else if (MatHead.Typ == 5) // 双精度浮点型
			{
				if (DataMajor == MatDataMajor::Row)
				{
					MatData.read(reinterpret_cast<char*>(Mat.data()), Row * Col * sizeof(double));
				}
				else if (DataMajor == MatDataMajor::Col)
				{
					double* MatBuff = new double[Num];
					MatData.read(reinterpret_cast<char*>(MatBuff), Num * sizeof(double));

					int Tol = static_cast<int>(Num);
#pragma omp parallel for schedule(static, 64) num_threads(8) if(Num > 8192)
					for (int k = 0; k < Tol; k++)
					{
						size_t m = static_cast<size_t>(k) / Col;
						size_t n = static_cast<size_t>(k) % Col;
						Mat[n * Row + m] = MatBuff[k];
					}
					delete[] MatBuff;
				}
				else
				{
					fprintf(stderr, "进入有关枚举类MatDataMajor的非法分支！");
					exit(EXIT_FAILURE);
				}
			}
			else
			{
				fprintf(stderr, "暂不支持该类型(%u)的矩阵数据读取！", MatHead.Typ);
				exit(EXIT_FAILURE);
			}
		}
		else
		{
			fprintf(stderr, "暂不支持该格式的矩阵数据读取！");
			exit(EXIT_FAILURE);
		}
	}
};

#endif
