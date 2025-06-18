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

enum class MatDataMajor : uint8_t { NaN = 0, Row = 1, Col = 2 }; // ����������ʽ��ö���࣬��NaN����ʾ��δ��ʼ��������Row����ʾ�������ȴ洢������Col����ʾ�������ȴ洢��
enum class MatFileFormat : uint8_t { UNDEFINED = 0, BMT = 1};    // �����ⲿ����������ݵĴ洢��ʽ��ö���࣬��BMT�������Զ����һ�ֲ��������ȴ洢�Ķ������ļ���ʽ

// BMT��ʽ�Ķ����ƾ��������ļ�������һ��16λ���ļ�ͷ
struct BMTHeader
{
	uint32_t Off; // ��������ʼλ�õ�λƫ��(�̶�ȡֵΪ16)
	uint32_t Typ; // ��������Ԫ�ص����ͣ�����4-�����ȸ�����(ÿ��Ԫ��4���ֽ�)��5-˫���ȸ�����(ÿ��Ԫ��8���ֽ�)[����ENVI��׼��https://www.nv5geospatialsoftware.com/docs/ENVIHeaderFiles.html]
	uint32_t Row; // ���������
	uint32_t Col; // ���������
};

// �Զ���ļ򵥾�����(ֻҪ��������������ݵı�׼������)
class matrix
{
private:
	size_t Row;              // ���������
	size_t Col;              // ���������
	MatDataMajor Maj;        // �������ʽ(�μ�MatDataMajorö����Ķ���)
	std::vector<double> Mat; // ����һά�������洢����Ԫ�أ���϶�άʵ���ܸ�����ؽ����ڴ渴��(�����ά�ǶԳ����ת�ñ������¿����ڴ�)

	bool resize(size_t row_new, size_t col_new, MatDataMajor maj_new) // ��������Ĵ�С
	{
		if (row_new == 0 || col_new == 0)
		{
			fprintf(stderr, "����ĳ�ʼ����С����Ϊ0��\n");
			return false;
		}
		if (maj_new == MatDataMajor::NaN)
		{
			fprintf(stderr, "����ʹ��δ����״̬��NaN������ʼ������\n");
			return false;
		}

		Row = row_new;
		Col = col_new;
		Maj = maj_new;
		Mat.resize(Row * Col);
		if (Mat.capacity() > Mat.size() + 128) Mat.shrink_to_fit(); // ��vector���������ϴ���ʵ�ʴ�Сʱ�����ͷŵ�������ڴ�
		return true;
	}

public:
	size_t row() const       // ���ؾ�������
	{
		return Row;
	}

	size_t col() const       // ���ؾ�������
	{
		return Col;
	}

	MatDataMajor maj() const // ���ؾ�����ʽ
	{
		return Maj;
	}

	void size(size_t& row_out, size_t& col_out) const // ͨ������ͬʱȡ�����������������
	{
		row_out = Row;
		col_out = Col;
	}

	void clear(bool ifShrinkMemory) // ��վ���ͨ����ifShrinkMemory�������������Ƿ�vector�����ڴ�Ҳ�ͷŵ�
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





	void LeastSquaresEstimation_Serial(const matrix& B, const matrix& l) // ��С���˹��ƵĴֲڴ���ʵ��
	{
		size_t obsv, para;  // ��obsv������۲���������para�������������
		B.size(obsv, para); // ʹ��matrix���size()����ȡ����ƾ���B��ά�ȡ����۲������Ͳ�������
		if (l.row() != obsv || l.col() != 1)
		{
			fprintf(stderr, "��ƾ���(%zu��%zu��)��۲�����(%zu��%zu��)�Ĵ�С��ƥ�䣡\n", obsv, para, l.row(), l.col());
			exit(EXIT_FAILURE);
		}

		if (Maj != MatDataMajor::NaN) clear(true);
		resize(para, 1, MatDataMajor::Col); // �������õ�ǰ������matrix�������ڴ��С�������洢�������Ҷ�����w�͹��ƽ��x(ά�Ⱦ�Ϊpara��1��)



		// ��һ��
		// ���㷨����ϵ������Nbb = B' * B���˴�����ʵ�ֲ��õ���ͬʱ�����������ȴ洢�������ȴ洢�ġ�()��������ء������ʾ���Ԫ��
		matrix Nbb(para, para, MatDataMajor::Row, 0.0);
#if defined(_CoutTiming_)
		std::cout << "���ֲڴ���ʵ�֡���ʱ��ʼ����\n";
		double T_stt = omp_get_wtime();
#endif
		for (size_t i = 0; i < para; i++)
		{
			for (size_t j = i; j < para; j++) // ����Nbb�ĶԳ��ԡ�ֻ����������ǲ���
			{
				for (size_t k = 0; k < obsv; k++) Nbb(i, j) += B(k, i) * B(k, j); // ����ֱ�ӵ�ַ���ʶ�������ʱ�ֲ�����ȡ��Ԫ�ؽ������㡢�ٽ���������ֵ��ȥ����������Ӽ��㿪��
				if (i != j) Nbb(j, i) = Nbb(i, j); // �����ǲ���ֱ�Ӹ��������ǵļ�����
			}
		}
#if defined(_CoutTiming_)
		double T_end = omp_get_wtime();
		double T_sub = T_end - T_stt;
		double T_sum = T_sub;
		std::cout << "���ֲڴ���ʵ�֡����ɷ�����ϵ�������ʱ" << T_sub << "��\n";
#endif



		// �ڶ���
		// ���㷨�����Ҷ�����w = B' * l���������洢�ڵ��õ�ǰ������matrix�������
#if defined(_CoutTiming_)
		T_stt = omp_get_wtime();
#endif
		for (size_t n = 0; n < para; n++)
		{
			(*this)(n, 0) = 0.0;
			for (size_t m = 0; m < obsv; m++) (*this)(n, 0) += B(m, n) * l(m, 0); // ��������ϵ������ļ������һ�������⣬����ͨ����ַ���ʶ�д�ڴ�ʹ�������ܵ���
		}
#if defined(_CoutTiming_)
		T_end = omp_get_wtime();
		T_sub = T_end - T_stt;
		T_sum += T_sub;
		std::cout << "���ֲڴ���ʵ�֡����ɷ������Ҷ�������ʱ" << T_sub << "��\n";
#endif



		// ������
		// �Է�����ϵ������Nbb���иĽ���Cholesky�ֽ⣬�õ�Nbb = L * D * L'������L�ǶԽ�Ԫ��Ϊ1�������Ǿ���D�ǶԽ�Ԫ������0�ĶԽǾ���
#if defined(_CoutTiming_)
		T_stt = omp_get_wtime();
#endif
		for (size_t q = 0; q < para; q++) // ���н���LDL'�ֽ�
		{
			for (size_t r = 0; r < q; r++) Nbb(q, q) -= Nbb(r, q) * Nbb(q, r) * Nbb(r, r); // ����ԽǾ���D�ĵ�q��Ԫ�ء�����洢��Nbb�ĶԽ�����

			for (size_t p = q + 1; p < para; p++) // ���������Ǿ���L�ĵ�q��Ԫ��
			{
				for (size_t s = 0; s < q; s++) Nbb(p, q) -= Nbb(p, s) * Nbb(s, s) * Nbb(s, q);
				Nbb(p, q) /= Nbb(q, q);
				Nbb(q, p) = Nbb(p, q); // Nbb�������ǲ��ּ�¼L�������ǲ��ּ�¼L'���ɶԳ��Կɵ������ǲ���ȡֵ
			}
		}
#if defined(_CoutTiming_)
		T_end = omp_get_wtime();
		T_sub = T_end - T_stt;
		T_sum += T_sub;
		std::cout << "���ֲڴ���ʵ�֡�������ϵ������ĸĽ�Cholesky�ֽ��ʱ" << T_sub << "��\n";
#endif



		// ���Ĳ�
		//��������ϵ������Nbb�ֽ�Ϊ�����Ǿ���L���ԽǾ���D�������Ǿ���L'֮�󣬲���ǰ�������б任(���ԶԽ�Ԫ)�ͻش�����ɷ����̵����
#if defined(_CoutTiming_)
		T_stt = omp_get_wtime();
#endif
		for (size_t u = 0; u < para - 1; u++) // ʹ��ǰ������ϵ������Ϊ�����Ǿ�������Է�����
		{
			for (size_t v = u + 1; v < para; v++) (*this)(v, 0) -= Nbb(v, u) * (*this)(u, 0);
		}

		for (size_t t = 0; t < para; t++) (*this)(t, 0) /= Nbb(t, t); // ����ԽǾ���ÿһ�г��Զ�Ӧ�Խ�Ԫ�صĵ���(�Ⱥ��Ҳ�Ϊ���������ÿһ��ֻ��һ��Ԫ����Ҫ����)

		for (size_t y = para - 1; y > 0; y--) // ʹ�ûش�����ϵ������Ϊ�����Ǿ�������Է�����
		{
			for (size_t x = 0; x < y; x++) (*this)(x, 0) -= Nbb(x, y) * (*this)(y, 0);
		}
#if defined(_CoutTiming_)
		T_end = omp_get_wtime();
		T_sub = T_end - T_stt;
		T_sum += T_sub;
		std::cout << "���ֲڴ���ʵ�֡����ڸĽ�Cholesky�ֽ�����Է���������ʱ" << T_sub << "��\n";
		std::cout << "���ֲڴ���ʵ�֡��ۼƹ���ʱ" << T_sum << "��\n\n";
#endif
	}





	void LeastSquaresEstimation_Parallel(const matrix& B, const matrix& l, bool on_off) // ��С���˹��Ƶ���Ѵ���ʵ�ּ���OpenMPֱ�Ӳ��л������С�on_off������OpenMP���л��Ŀ���
	{
		size_t obsv, para;
		B.size(obsv, para);
		if (B.maj() != MatDataMajor::Col) // Nbb = B' * B�ĵ�i�С���j��Ԫ�ؼ�����B�ĵ�i�����j�е��ڻ��������������ȴ洢ʱ����B��ÿһ�ж��ڴ����������������ʴ������
		{
			fprintf(stderr, "���벢����С���˵���ƾ��������������ȴ洢��\n");
			exit(EXIT_FAILURE);
		}
		if (l.row() != obsv || l.col() != 1)
		{
			fprintf(stderr, "��ƾ���(%zu��%zu��)��۲�����(%zu��%zu��)�Ĵ�С��ƥ�䣡\n", obsv, para, l.row(), l.col());
			exit(EXIT_FAILURE);
		}

		if (Maj != MatDataMajor::NaN) clear(true);
		resize(para, 1, MatDataMajor::Col); // �������øú�����matrix�������ڴ��С�������洢�������Ҷ�����w�͹��ƽ��x(ά�Ⱦ�Ϊpara��1��)
		


		// ��һ��
		// ���㷨����ϵ������Nbb = B' * B
		matrix Nbb(para, para, MatDataMajor::Row, 0.0);
#if defined(_CoutTiming_)
		if (on_off) std::cout << "��OpenMP����ʵ�֡���ʱ��ʼ����\n";
		else std::cout << "����Ѵ���ʵ�֡���ʱ��ʼ����\n";
		double T_stt = omp_get_wtime();
#endif
		const int Wid = static_cast<int>(para);
		const int Hgt = static_cast<int>(obsv);
#pragma omp parallel for schedule(dynamic, 4) num_threads(16) if(on_off)    // ÿ���̸߳���Nbb��4��Ԫ�صļ��㣬����Nbb�ĵ�i��������Ԫ������Ϊpara - i�����̸߳��ر�Ȼ��ͬ���ʲ���dynamic��̬����
		for (int i = 0; i < Wid; i++)
		{
			const double* pCol_i = B.Mat.data() + i * Hgt;                  // ��ƾ���B��i�е���ָ��
			for (int j = i; j < Wid; j++)
			{
				const double* pCol_j = B.Mat.data() + j * Hgt;              // ��ƾ���B��j�е���ָ��

				double Inp = 0.0;                                           // ʹ�þֲ���ʱ�����洢�ڻ����
				for (int k = 0; k < Hgt; k++) Inp += pCol_i[k] * pCol_j[k]; // ������ƾ���B�ĵ�i�к͵�j�е��ڻ�����
				Nbb.Mat[i * Wid + j] = Inp;                                 // ���������洢������Nbb��
				if (i != j) Nbb.Mat[j * Wid + i] = Inp;                     // ���¶ԳƲ�Ԫ��
			}
		}
#if defined(_CoutTiming_)
		double T_end = omp_get_wtime();
		double T_sub = T_end - T_stt;
		double T_sum = T_sub;
		if (on_off) std::cout << "��OpenMP����ʵ�֡����ɷ�����ϵ�������ʱ" << T_sub << "��\n";
		else std::cout << "����Ѵ���ʵ�֡����ɷ�����ϵ�������ʱ" << T_sub << "��\n";
#endif



		// �ڶ���
		// ���㷨�����Ҷ�����w = B' * l
#if defined(_CoutTiming_)
		T_stt = omp_get_wtime();
#endif
		const double* pCol_l = l.Mat.data();                            // �۲�����l����ָ��
#pragma omp parallel for schedule(static) num_threads(16) if(on_off)
		for (int p = 0; p < Wid; p++)
		{
			const double* pCol_b = B.Mat.data() + p * Hgt;              // ��ƾ���B��p�е���ָ��

			double Inp = 0.0;                                           // ʹ�þֲ���ʱ�����洢�ڻ����
			for (int q = 0; q < Hgt; q++) Inp += pCol_b[q] * pCol_l[q]; // ������ƾ���B�ĵ�p�к͹۲�����l���ڻ�����
			(*this).Mat[p] = Inp;                                       // ���������洢�����øú����ľ�����
		}
#if defined(_CoutTiming_)
		T_end = omp_get_wtime();
		T_sub = T_end - T_stt;
		T_sum += T_sub;
		if (on_off) std::cout << "��OpenMP����ʵ�֡����ɷ������Ҷ�������ʱ" << T_sub << "��\n";
		else std::cout << "����Ѵ���ʵ�֡����ɷ������Ҷ�������ʱ" << T_sub << "��\n";
#endif



		// ������
		// �Է�����ϵ������Nbb����LDL'�ֽ�
#if defined(_CoutTiming_)
		T_stt = omp_get_wtime();
#endif
		std::vector<double> D(para, 0.0); // ʹ��һ���������ʱ�����������洢�ԽǾ���D�ĶԽ�Ԫ�أ��Ա��ں���ʹ��ʱ��߻���������
		const double* pVec_d = D.data();  // �Գƾ���D�ĶԽ�Ԫ����������ָ��
		for (int r = 0; r < Wid; r++)
		{
			// ���ȼ���ԽǾ���D�ĵ�r���Խ�Ԫ
			double subt = 0.0;
			const double* pRow_r = Nbb.Mat.data() + r * Wid;                       // ����Nbb��r�е���ָ�룬�˴�ʵ�ʷ��ʵ��Ǵ洢�ھ���Nbb�е�ǰ���ηֽ���
			for (int s = 0; s < r; s++) subt += pRow_r[s] * pRow_r[s] * pVec_d[s];
			double diag = Nbb.Mat[r * Wid + r] - subt;                             // ʹ�þֲ���ʱ�����洢��r���Խ�Ԫ���������Ա���������������Ǿ���L�ĵ�r��Ԫ�ص�ʹ��
			Nbb.Mat[r * Wid + r] = 1.0;                                            // �����Ǿ���L�ĶԽ�Ԫ��Ϊ1(�ò���ʵ�ʶ��ࡢֻΪ���������Ǿ���L������)
			D[r] = diag;                                                           // ���Խ�Ԫ�ļ����������浽����D��

			if (diag < 0.0 || std::fabs(diag) < 1e-12)
			{
				fprintf(stderr, "������ϵ���������죬�޷������С���˽��㣡");
				exit(EXIT_FAILURE);
			}

			// ��μ��������Ǿ���L�ĵ�r��Ԫ�أ��ò���ֻ��ǰpara - 1��(����Ϊ0 ~ para - 2��para = Wid)���У���Ϊ���һ��ֻ�жԽ�Ԫ���Խ�������û��Ԫ��
			if (r < Wid - 1)
			{
#pragma omp parallel for schedule(static) num_threads(8) if (r < Wid - 32 && on_off)  // ��r >= Wid - 32ʱ����r�е�������Ԫ��С��32�������̷߳��䵽����������4������ʱ���̵߳��ȿ��������ڲ��л�����
				for (int u = r + 1; u < Wid; u++)
				{
					double subt = 0.0;
					const double* pRow_u = Nbb.Mat.data() + u * Wid;
					for (int v = 0; v < r; v++) subt += pRow_r[v] * pRow_u[v] * D[v]; // �ۼ���ֻ��r�йء�����r�и�������Ԫ�صļ�����һ�£���˲���static��̬���ȼ��ɱ�֤���ؾ���

					double elem = Nbb.Mat[r * Wid + u];
					Nbb.Mat[r * Wid + u] = (elem - subt) / diag;
					Nbb.Mat[u * Wid + r] = (elem - subt) / diag; // ͬʱ���¶ԳƲ��֣��Ա��ں���ǰ�����ͻش��������Է������ʹ��
				}
			}
		}
#if defined(_CoutTiming_)
		T_end = omp_get_wtime();
		T_sub = T_end - T_stt;
		T_sum += T_sub;
		if (on_off) std::cout << "��OpenMP����ʵ�֡�������ϵ������ĸĽ�Cholesky�ֽ��ʱ" << T_sub << "��\n";
		else std::cout << "����Ѵ���ʵ�֡�������ϵ������ĸĽ�Cholesky�ֽ��ʱ" << T_sub << "��\n";
#endif



		// ���Ĳ�
		// ���Է�����Nbb * x = w����ɾ���ֽ�֮��ת��Ϊ (L * D * L') * x = w������LΪ�Խ�Ԫ��Ϊ1�������Ǿ���DΪ�Խ�Ԫ������0�ĶԽǾ���ֻ�����β���ǰ�������б任(ÿһ�г��ԶԽ���D��λ����ͬ�еĶԽ�Ԫ��)�ͻش������ɿ���������Է���������
#if defined(_CoutTiming_)
		T_stt = omp_get_wtime();
#endif
		// ǰ��������ʱNbb�������ǲ��ִ洢��L'�������ǲ��ִ洢��L��ǰ����ʹ�õ����������ǵ�L������Nbb���������ȴ洢����ǰ������Ҫ���з�����ʣ��Դ˿�ͨ�����з�����������ǵ�L'��������Ӷ���������������(��Ϊ���з����ڴ�����)
		double* pVec_w = (*this).Mat.data();                                   // �������Ҷ�����w����ָ��
		for (int m = 0; m < Wid - 1; m++)
		{
			const double val_Wm = (*this).Mat[m];                              // ȡ��Wm
			const double* pUpp_m = Nbb.Mat.data() + m * Wid;                   // ����Nbb��m�е���ָ�롢�˴��൱�������Ǿ���L��m�е���ָ��
			for (int n = m + 1; n < Wid; n++) pVec_w[n] -= val_Wm * pUpp_m[n]; // Wn -= Wm * Lnm
		}

		// �б任
		for (int h = 0; h < Wid; h++) pVec_w[h] /= pVec_d[h];                  // Wh /= Dhh

		// �ش������ش���ʹ�õ����������ǵ�L'�����з�����з��ʣ���ǰ������ʵ�����ơ�ͨ�����з����ȡNbb�����ǵ�L����֤�ڴ������ͻ���������
		for (int f = Wid - 1; f > 0; f--)
		{
			const double val_Wf = (*this).Mat[f];                              // ȡ��Wf
			const double* pLow_f = Nbb.Mat.data() + f * Wid;                   // ����Nbb��f�е���ָ�롢�˴��൱�������Ǿ���L'��f�е���ָ��
			for (int g = 0; g < f; g++) pVec_w[g] -= val_Wf * pLow_f[g];       // Wg -= Wf * L'gf
		}
#if defined(_CoutTiming_)
		T_end = omp_get_wtime();
		T_sub = T_end - T_stt;
		T_sum += T_sub;
		if (on_off)
		{
			std::cout << "��OpenMP����ʵ�֡����ڸĽ�Cholesky�ֽ�����Է���������ʱ" << T_sub << "��\n";
			std::cout << "��OpenMP����ʵ�֡��ۼƹ���ʱ" << T_sum << "��\n\n";
		}
		else
		{
			std::cout << "����Ѵ���ʵ�֡����ڸĽ�Cholesky�ֽ�����Է���������ʱ" << T_sub << "��\n";
			std::cout << "����Ѵ���ʵ�֡��ۼƹ���ʱ" << T_sum << "��\n\n";
		}
#endif
	}





	void LeastSquaresEstimation_Parallel_AVX(const matrix& B, const matrix& l) // ��OpenMPֱ�Ӳ��л��Ļ����ϣ�ͨ��������������������߼���Ч�ʣ��������������������ʵ�ֽ��н��ܡ���ģ���ʵ�������LeastSquaresEstimation_Parallel()�������ע�ͺ�ʵϰ����
	{
		size_t obsv, para;
		B.size(obsv, para);
		if (B.maj() != MatDataMajor::Col)
		{
			fprintf(stderr, "���벢����С���˵���ƾ��������������ȴ洢��\n");
			exit(EXIT_FAILURE);
		}
		if (l.row() != obsv || l.col() != 1)
		{
			fprintf(stderr, "��ƾ���(%zu��%zu��)��۲�����(%zu��%zu��)�Ĵ�С��ƥ�䣡\n", obsv, para, l.row(), l.col());
			exit(EXIT_FAILURE);
		}

		if (Maj != MatDataMajor::NaN) clear(true);
		resize(para, 1, MatDataMajor::Col);



		// ��һ������
		// ���㷨���̵�ϵ������Nbb = B' * B���Ҷ�����w = B' * l
		matrix Nbb(para, para, MatDataMajor::Row, 0.0);
#if defined(_CoutTiming_)
		std::cout << "��OpenMP(AVX2)����ʵ�֡���ʱ��ʼ����\n";
		double T_stt = omp_get_wtime();
#endif
		const int Wid = static_cast<int>(para);
		const int Hgt = static_cast<int>(obsv);
		const int Tol = static_cast<int>(obsv) - 16;
		const double* pCol_l = l.Mat.data();

#pragma omp parallel num_threads(16)
		{
			__m256d accu_1, accu_2, accu_3, accu_4; // ����ÿ���߳�˽�е��ۼ�����ÿ���Ĵ�����ͬʱ��4��˫���ȸ��������ݽ��м��㡢���ĸ��Ĵ������Ӧ16��˫���ȸ���������

			// ���㷨���̵�ϵ������Nbb
#pragma omp for schedule(dynamic, 4)
			for (int i = 0; i < Wid; i++)
			{
				const double* pCol_i = B.Mat.data() + i * Hgt;
				for (int j = i; j < Wid; j++)
				{
					const double* pCol_j = B.Mat.data() + j * Hgt;

					// �ۼ���������ʼ��
					accu_1 = _mm256_setzero_pd();
					accu_2 = _mm256_setzero_pd();
					accu_3 = _mm256_setzero_pd();
					accu_4 = _mm256_setzero_pd();

					int k = 0;
					for (; k <= Tol; k += 16) // ÿ��ȡ16��˫���ȸ���������
					{
						// ��ǰ����һ��Ҫ���ʵ����ݼ��ص�L1���桢�Լ����򻺴治������ɵ��ӳ�
						_mm_prefetch(reinterpret_cast<const char*>(pCol_i + k + 16), _MM_HINT_T0);
						_mm_prefetch(reinterpret_cast<const char*>(pCol_j + k + 16), _MM_HINT_T0);

						// ÿ������ȡ��4��˫���ȸ����������Խ��оֲ��ڻ�����
						__m256d vec_i1 = _mm256_loadu_pd(pCol_i + k);
						__m256d vec_i2 = _mm256_loadu_pd(pCol_i + k + 4);
						__m256d vec_i3 = _mm256_loadu_pd(pCol_i + k + 8);
						__m256d vec_i4 = _mm256_loadu_pd(pCol_i + k + 12);

						__m256d vec_j1 = _mm256_loadu_pd(pCol_j + k);
						__m256d vec_j2 = _mm256_loadu_pd(pCol_j + k + 4);
						__m256d vec_j3 = _mm256_loadu_pd(pCol_j + k + 8);
						__m256d vec_j4 = _mm256_loadu_pd(pCol_j + k + 12);

						// ͨ�����ںϳ˼ӡ��������ֲ�(4��)�ڻ�������µ��ۼ�����
						accu_1 = _mm256_fmadd_pd(vec_i1, vec_j1, accu_1);
						accu_2 = _mm256_fmadd_pd(vec_i2, vec_j2, accu_2);
						accu_3 = _mm256_fmadd_pd(vec_i3, vec_j3, accu_3);
						accu_4 = _mm256_fmadd_pd(vec_i4, vec_j4, accu_4);
					}
					// ����ܱ�16�������ֵ��ڻ�����֮�󡢾ۺϸ��ۼ����ļ�����
					accu_1 = _mm256_add_pd(accu_1, accu_2);
					accu_3 = _mm256_add_pd(accu_3, accu_4);
					accu_1 = _mm256_add_pd(accu_1, accu_3);

					__m128d vec_2l = _mm256_castpd256_pd128(accu_1);           // ��ȡ��128λ(2��˫���ȸ���������)��ͨ��ֱ������ת��ʵ�֡���ϡ�_mm256_extractf128_pd(accu_1, 0)����������Ч�ʸ���
					                                                           // accu_1: [a, b, c, d];   vec_2l: [a, b]
					__m128d vec_2h = _mm256_extractf128_pd(accu_1, 1);         // ��ȡ��128λ(2��˫���ȸ���������)
					                                                           // accu_1: [a, b, c, d];   vec_2h: [c, d]
					vec_2l = _mm_add_pd(vec_2l, vec_2h);                       // ������128λ�������
					                                                           // vec_2l: [a + c, b + d]; vec_2h: [c, d]
					vec_2h = _mm_unpackhi_pd(vec_2l, vec_2l);                  // ������128λ������ȡ���Եĸ�64λ����µ�128λ�������˴�vec_2h������64λ���洢vec_2l�ĸ�64λ
					                                                           // vec_2l: [a + c, b + d]; vec_2h: [b + d, b + d]
					double result = _mm_cvtsd_f64(_mm_add_sd(vec_2l, vec_2h)); // vec_2l��vec_2h��ӽ���ĵ�64λ����vec_2l�ĸ�64λ�͵�64λ�ĺͣ���_mm_cvtsd_f64����128λ�����ĵ�64λת��Ϊ˫���ȸ�����
					                                                           // vec_2l + vec_2h: [a + b + c + d, b + b + d + d] -> return: [a + b + c + d]

					for (; k < Hgt; k++) result += pCol_i[k] * pCol_j[k];      // ���ʣ�ಿ�ֵ��ڻ�����
					Nbb.Mat[i * Wid + j] = result;                             // ��������浽����Nbb��
					if (i != j) Nbb.Mat[j * Wid + i] = result;                 // ͬʱ����ԳƲ���
				}
			}

			// ���㷨���̵��Ҷ�����w
#pragma omp for schedule(static)
			for (int p = 0; p < Wid; p++)
			{
				const double* pCol_b = B.Mat.data() + p * Hgt;

				// �ۼ���������ʼ��
				accu_1 = _mm256_setzero_pd();
				accu_2 = _mm256_setzero_pd();
				accu_3 = _mm256_setzero_pd();
				accu_4 = _mm256_setzero_pd();

				int q = 0;
				for (; q <= Tol; q += 16) // ÿ��ȡ16��˫���ȸ���������
				{
					// ��ǰ����һ��Ҫ���ʵ����ݼ��ص�L1���桢�Լ����򻺴治������ɵ��ӳ�
					_mm_prefetch(reinterpret_cast<const char*>(pCol_b + q + 16), _MM_HINT_T0);
					_mm_prefetch(reinterpret_cast<const char*>(pCol_l + q + 16), _MM_HINT_T0);

					// ÿ������ȡ��4��˫���ȸ����������Խ��оֲ��ڻ�����
					__m256d vec_b1 = _mm256_loadu_pd(pCol_b + q);
					__m256d vec_b2 = _mm256_loadu_pd(pCol_b + q + 4);
					__m256d vec_b3 = _mm256_loadu_pd(pCol_b + q + 8);
					__m256d vec_b4 = _mm256_loadu_pd(pCol_b + q + 12);

					__m256d vec_l1 = _mm256_loadu_pd(pCol_l + q);
					__m256d vec_l2 = _mm256_loadu_pd(pCol_l + q + 4);
					__m256d vec_l3 = _mm256_loadu_pd(pCol_l + q + 8);
					__m256d vec_l4 = _mm256_loadu_pd(pCol_l + q + 12);

					// ͨ�����ںϳ˼ӡ��������ֲ�(4��)�ڻ�������µ��ۼ�����
					accu_1 = _mm256_fmadd_pd(vec_b1, vec_l1, accu_1);
					accu_2 = _mm256_fmadd_pd(vec_b2, vec_l2, accu_2);
					accu_3 = _mm256_fmadd_pd(vec_b3, vec_l3, accu_3);
					accu_4 = _mm256_fmadd_pd(vec_b4, vec_l4, accu_4);
				}
				// ����ܱ�16�������ֵ��ڻ�����֮�󡢾ۺϸ��ۼ����ļ�����
				accu_1 = _mm256_add_pd(accu_1, accu_2);
				accu_3 = _mm256_add_pd(accu_3, accu_4);
				accu_1 = _mm256_add_pd(accu_1, accu_3);

				__m128d vec_2l = _mm256_castpd256_pd128(accu_1);           // ��ȡ��128λ(2��˫���ȸ���������)��ͨ��ֱ������ת��ʵ�֡���ϡ�_mm256_extractf128_pd(accu_1, 0)����������Ч�ʸ���
				                                                           // accu_1: [a, b, c, d];   vec_2l: [a, b]
				__m128d vec_2h = _mm256_extractf128_pd(accu_1, 1);         // ��ȡ��128λ(2��˫���ȸ���������)
				                                                           // accu_1: [a, b, c, d];   vec_2h: [c, d]
				vec_2l = _mm_add_pd(vec_2l, vec_2h);                       // ������128λ�������
				                                                           // vec_2l: [a + c, b + d]; vec_2h: [c, d]
				vec_2h = _mm_unpackhi_pd(vec_2l, vec_2l);                  // ������128λ������ȡ���Եĸ�64λ����µ�128λ�������˴�vec_2h������64λ���洢vec_2l�ĸ�64λ
				                                                           // vec_2l: [a + c, b + d]; vec_2h: [b + d, b + d]
				double result = _mm_cvtsd_f64(_mm_add_sd(vec_2l, vec_2h)); // vec_2l��vec_2h��ӽ���ĵ�64λ����vec_2l�ĸ�64λ�͵�64λ�ĺͣ���_mm_cvtsd_f64����128λ�����ĵ�64λת��Ϊ˫���ȸ�����
				                                                           // vec_2l + vec_2h: [a + b + c + d, b + b + d + d] -> return: [a + b + c + d]

				for (; q < Hgt; q++) result += pCol_b[q] * pCol_l[q];      // ���ʣ�ಿ�ֵ��ڻ�����
				(*this).Mat[p] = result;                                   // ��������浽���øú����ľ�����
			}
		}
#if defined(_CoutTiming_)
		double T_end = omp_get_wtime();
		double T_sub = T_end - T_stt;
		double T_sum = T_sub;
		std::cout << "��OpenMP(AVX2)����ʵ�֡����ɷ����̹���ʱ" << T_sub << "��\n";
#endif



		// ������
		// �����̵�ϵ������Nbb�ĸĽ�Cholesky�ֽ�
#if defined(_CoutTiming_)
		T_stt = omp_get_wtime();
#endif
		std::vector<double> D(para, 0.0);
		const double* pVec_d = D.data();
		for (int r = 0; r < Wid; r++)
		{
			// ���ȼ���ԽǾ���D�ĵ�r���Խ�Ԫ
			double diag = Nbb.Mat[r * Wid + r];
			const double* pRow_r = Nbb.Mat.data() + r * Wid;
			if (r < 4) // ǰ4��������ۼӼ���Ĵ���С��4������ʹ��������
			{
				for (int s = 0; s < r; s++) diag -= Nbb.Mat[r * Wid + s] * Nbb.Mat[r * Wid + s] * D[s];
				D[r] = diag;
			}
			else
			{
				__m256d accu_d = _mm256_setzero_pd();               //��������ʼ���ۼ���

				int s = 0;
				for (; s <= r - 4; s += 4)                          // ÿ��ȡ4��˫���ȸ��������ݽ��м���
				{
					__m256d vec_rs = _mm256_loadu_pd(pRow_r + s);   // ȡ��Lrs ~ Lr(s+3)
					__m256d vec_ds = _mm256_loadu_pd(pVec_d + s);   // ȡ��Dss ~ D(s+3)(s+3)
					__m256d mul_rd = _mm256_mul_pd(vec_rs, vec_rs); // ����Lri * Lri * Dii������i = s ~ s+3
					mul_rd = _mm256_mul_pd(mul_rd, vec_ds);
					accu_d = _mm256_add_pd(accu_d, mul_rd);         // �ۼӼ�����
				}

				// ͬ�Ͻ�256λ�����е�4��˫���ȸ����������ۼӺʹ���
				__m128d vec_2l = _mm256_castpd256_pd128(accu_d);
				__m128d vec_2h = _mm256_extractf128_pd(accu_d, 1);
				vec_2l = _mm_add_pd(vec_2l, vec_2h);
				vec_2h = _mm_unpackhi_pd(vec_2l, vec_2l);
				diag -= _mm_cvtsd_f64(_mm_add_sd(vec_2l, vec_2h));

				// ���ʣ�ಿ�ֵļ���
				for (; s < r; s++) diag -= Nbb.Mat[r * Wid + s] * Nbb.Mat[r * Wid + s] * D[s];
				D[r] = diag;
			}
			Nbb.Mat[r * Wid + r] = 1.0;

			if (diag < 0.0 || std::fabs(diag) < 1e-12)
			{
				fprintf(stderr, "������ϵ���������죬�޷������С���˽��㣡");
				exit(EXIT_FAILURE);
			}

			// ��μ��������Ǿ���L�ĵ�r��Ԫ��
			if (r < Wid - 1)
			{
				if (r < 16) // ǰ��������ʹ�õ����������򵥲����Ż�����
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
						__m256d accu_o; // �������߳�˽�е��ۼ���

#pragma omp for schedule(static)
						for (int u = r + 1; u < Wid; u++)
						{
							accu_o = _mm256_setzero_pd(); // ��ʼ���ۼ���

							double elem = Nbb.Mat[r * Wid + u];
							const double* pRow_u = Nbb.Mat.data() + u * Wid;

							int v = 0;
							for (; v <= r - 4; v += 4)
							{
								// ����Luv * Lvr * Dvv������v�����ۼ�(���ڶԳ�����Luv = Lvu��Lvr = Lrv)
								__m256d vec_rv = _mm256_loadu_pd(pRow_r + v);
								__m256d vec_uv = _mm256_loadu_pd(pRow_u + v);
								__m256d vec_dv = _mm256_loadu_pd(pVec_d + v);
								__m256d mul_rd = _mm256_mul_pd(vec_rv, vec_uv);
								mul_rd = _mm256_mul_pd(mul_rd, vec_dv);
								accu_o = _mm256_add_pd(accu_o, mul_rd);
							}

							// ͬ�Ͻ�256λ�����е�4��˫���ȸ����������ۼӺʹ���
							__m128d vec_2l = _mm256_castpd256_pd128(accu_o);
							__m128d vec_2h = _mm256_extractf128_pd(accu_o, 1);
							vec_2l = _mm_add_pd(vec_2l, vec_2h);
							vec_2h = _mm_unpackhi_pd(vec_2l, vec_2l);
							elem -= _mm_cvtsd_f64(_mm_add_sd(vec_2l, vec_2h));

							// ���ʣ�ಿ�ֵļ���
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
		std::cout << "��OpenMP(AVX2)����ʵ�֡�������ϵ������ĸĽ�Cholesky�ֽ��ʱ" << T_sub << "��\n";
#endif
		


		// ���Ĳ�
		// ������Է�����Nbb * x = (L * D * L') * x = w
#if defined(_CoutTiming_)
		T_stt = omp_get_wtime();
#endif
		// ǰ��������ǰ���������з�����������ǵ�L'���������з�����������ǵ�L
		double* pVec_w = (*this).Mat.data();
		for (int m = 0; m < Wid - 1; m++)
		{
			const double val_Wm = (*this).Mat[m];
			const double* pUpp_m = Nbb.Mat.data() + m * Wid;
			if (m < Wid - 4)
			{
				__m256d vec_Wm = _mm256_set1_pd(val_Wm);

				int n = m + 1;
				for (; n <= Wid - 4; n += 4) // ��������Wn -= Wm * Lnm
				{
					__m256d vec_Ln = _mm256_loadu_pd(pUpp_m + n);
					__m256d vec_Wn = _mm256_loadu_pd(pVec_w + n);
					vec_Ln = _mm256_mul_pd(vec_Ln, vec_Wm);
					vec_Wn = _mm256_sub_pd(vec_Wn, vec_Ln);
					_mm256_storeu_pd(pVec_w + n, vec_Wn);
				}
				for (; n < Wid; n++) pVec_w[n] -= val_Wm * pUpp_m[n];
			}
			else // �������ʣ��Ԫ�����������Խ�����������ֱ����ͨ����ʵ��
			{
				for (int n = m + 1; n < Wid; n++) pVec_w[n] -= val_Wm * pUpp_m[n];
			}
		}

		// �б任��ʵ�ַ������Ҷ�����w�ͶԽǾ���D�ĶԽ�Ԫ�ص�������
		int h = 0;
		for (; h <= Wid - 4; h += 4)
		{
			__m256d vec_Dh = _mm256_loadu_pd(pVec_d + h);
			__m256d vec_Wh = _mm256_loadu_pd(pVec_w + h);
			vec_Wh = _mm256_div_pd(vec_Wh, vec_Dh);
			_mm256_storeu_pd(pVec_w + h, vec_Wh);
		}
		for (; h < Wid; h++) pVec_w[h] /= pVec_d[h];

		// �ش�������ǰ���������з�����������ǵ�L���������з�����������ǵ�L'
		for (int f = Wid - 1; f > 0; f--)
		{
			const double val_Wf = (*this).Mat[f];
			const double* pLow_f = Nbb.Mat.data() + f * Wid;
			if (f > 3)
			{
				__m256d vec_Wf = _mm256_set1_pd(val_Wf);

				int g = 0;
				for (; g <= f - 4; g += 4) // ��������Wg -= Wf * L'gf
				{
					__m256d vec_Lg = _mm256_loadu_pd(pLow_f + g);
					__m256d vec_Wg = _mm256_loadu_pd(pVec_w + g);
					vec_Lg = _mm256_mul_pd(vec_Lg, vec_Wf);
					vec_Wg = _mm256_sub_pd(vec_Wg, vec_Lg);
					_mm256_storeu_pd(pVec_w + g, vec_Wg);
				}
				for (; g < f; g++) pVec_w[g] -= val_Wf * pLow_f[g];
			}
			else // �������(λ�������Ǿ���L'����ࡢ�ش����Ǵ���������е�)ʣ��Ԫ�����������Խ�����������ֱ����ͨ����ʵ��
			{
				for (int g = 0; g < f; g++) pVec_w[g] -= val_Wf * pLow_f[g];
			}
		}
#if defined(_CoutTiming_)
		T_end = omp_get_wtime();
		T_sub = T_end - T_stt;
		T_sum += T_sub;
		std::cout << "��OpenMP(AVX2)����ʵ�֡����ڸĽ�Cholesky�ֽ�����Է���������ʱ" << T_sub << "��\n";
		std::cout << "��OpenMP(AVX2)����ʵ�֡��ۼƹ���ʱ" << T_sum << "��\n\n";
#endif
	}





	double& operator()(size_t i, size_t j)
	{
		if (Maj == MatDataMajor::NaN)
		{
			fprintf(stderr, "��ǰ����Ϊ�գ�");
			exit(EXIT_FAILURE);
		}

		if (i >= Row || j >= Col)
		{
			fprintf(stderr, "�±�����������Χ��");
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
			fprintf(stderr, "�����й�ö����MatDataMajor�ķǷ���֧��");
			exit(EXIT_FAILURE);
		}
	}



	const double& operator()(size_t i, size_t j) const
	{
		if (Maj == MatDataMajor::NaN)
		{
			fprintf(stderr, "��ǰ����Ϊ�գ�");
			exit(EXIT_FAILURE);
		}

		if (i >= Row || j >= Col)
		{
			fprintf(stderr, "�±�����������Χ��");
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
			fprintf(stderr, "�����й�ö����MatDataMajor�ķǷ���֧��");
			exit(EXIT_FAILURE);
		}
	}



	matrix() : Row(0), Col(0), Maj(MatDataMajor::NaN), Mat() {}



	matrix(size_t row_new, size_t col_new, MatDataMajor maj_new)
	{
		if (row_new == 0 || col_new == 0)
		{
			fprintf(stderr, "����ĳ�ʼ����С����Ϊ0��\n");
			exit(EXIT_FAILURE);
		}
		if (maj_new == MatDataMajor::NaN)
		{
			fprintf(stderr, "����ʹ��δ����״̬��NaN������ʼ������\n");
			exit(EXIT_FAILURE);
		}

		Row = row_new;
		Col = col_new;
		Maj = maj_new;
		Mat.resize(Row * Col);
	}



	matrix(size_t row_new, size_t col_new, MatDataMajor maj_new, double val_ini) // ��val_ini���ǳ�ʼ����������ֵ
	{
		if (row_new == 0 || col_new == 0)
		{
			fprintf(stderr, "����ĳ�ʼ����С����Ϊ0��\n");
			exit(EXIT_FAILURE);
		}
		if (maj_new == MatDataMajor::NaN)
		{
			fprintf(stderr, "����ʹ��δ����״̬��NaN������ʼ������\n");
			exit(EXIT_FAILURE);
		}

		Row = row_new;
		Col = col_new;
		Maj = maj_new;
		Mat.resize(Row * Col, val_ini);
	}



	matrix(const std::string& FilePath, MatFileFormat FileFormat, MatDataMajor DataMajor) // ���ڶ�ȡ��BMT�������Ƹ�ʽ�������ݵĹ��캯��
	{
		if (DataMajor == MatDataMajor::NaN)
		{
			fprintf(stderr, "����ʹ��δ����״̬��NaN������ʼ������\n");
			exit(EXIT_FAILURE);
		}
		Maj = DataMajor;

		if (FileFormat == MatFileFormat::BMT)
		{
			std::ifstream MatData(FilePath, std::ios::binary);
			if (!MatData.is_open())
			{
				fprintf(stderr, "�޷��򿪸þ��������ļ���%s��\n", FilePath.c_str());
				exit(EXIT_FAILURE);
			}

			BMTHeader MatHead = {};
			MatData.read(reinterpret_cast<char*>(&MatHead), sizeof(MatHead));
			if (MatHead.Off != 16)
			{
				fprintf(stderr, "�ļ�ͷ������BMT��ʽ������\n");
				exit(EXIT_FAILURE);
			}
			Row = static_cast<size_t>(MatHead.Row);
			Col = static_cast<size_t>(MatHead.Col);
			size_t Num = Row * Col;
			Mat.resize(Num);

			if (MatHead.Typ == 4) // �����ȸ�����
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
					fprintf(stderr, "�����й�ö����MatDataMajor�ķǷ���֧��");
					exit(EXIT_FAILURE);
				}
				delete[] MatBuff;
			}
			else if (MatHead.Typ == 5) // ˫���ȸ�����
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
					fprintf(stderr, "�����й�ö����MatDataMajor�ķǷ���֧��");
					exit(EXIT_FAILURE);
				}
			}
			else
			{
				fprintf(stderr, "�ݲ�֧�ָ�����(%u)�ľ������ݶ�ȡ��", MatHead.Typ);
				exit(EXIT_FAILURE);
			}
		}
		else
		{
			fprintf(stderr, "�ݲ�֧�ָø�ʽ�ľ������ݶ�ȡ��");
			exit(EXIT_FAILURE);
		}
	}
};

#endif
