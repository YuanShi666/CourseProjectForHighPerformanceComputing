#include <cmath>
#include <string>
#include <iostream>

#include <omp.h>

#define _CoutTiming_ //若定义则启动逐步骤的计时输出，不过计时和输出操作可能会影响到总耗时
#include "BMT.h"

using namespace std;

void CoutResult(const matrix& Ref, const matrix& Res, double Stt, double End)
{
	cout << "总共耗时(秒)：" << End - Stt << "\n";
	double rel = 0.0;
	for (int i = 0; i < Ref.row(); i++)
	{
		double tmp = abs(1.0 - Res(i, 0) / Ref(i, 0));
		if (tmp > rel) rel = tmp;
	}
	cout << "最大相对误差：" << rel << "\n\n\n\n";
}

int main()
{
	//应保证设计矩阵、观测向量和解向量在维度上的适配
	string FileName = "../DataSet/DesMat_10240_1024.bmt";
	matrix DesMat(FileName, MatFileFormat::BMT, MatDataMajor::Col);

	FileName = "../DataSet/ObsVec_10240.bmt";
	matrix ObsVec(FileName, MatFileFormat::BMT, MatDataMajor::Col);

	FileName = "../DataSet/SolVec_1024.bmt";
	matrix SolRef(FileName, MatFileFormat::BMT, MatDataMajor::Col);

	double stt, end;
	//快速(取消)注释快捷键：Ctrl+K+C & Ctrl+K+U

	//matrix SolRaw;
	//stt = omp_get_wtime();
	//SolRaw.LeastSquaresEstimation_Serial(DesMat, ObsVec);          //粗糙串行实现(运行速度较慢、不建议调用)
	//end = omp_get_wtime();
	//CoutResult(SolRef, SolRaw, stt, end);

	matrix SolSer;
	stt = omp_get_wtime();
	SolSer.LeastSquaresEstimation_Parallel(DesMat, ObsVec, false); //最佳串行实现(不执行并行化的“OpenMP并行实现”)
	end = omp_get_wtime();
	CoutResult(SolRef, SolSer, stt, end);

	matrix SolPar;
	stt = omp_get_wtime();
	SolPar.LeastSquaresEstimation_Parallel(DesMat, ObsVec, true);  //OpenMP并行实现
	end = omp_get_wtime();
	CoutResult(SolRef, SolPar, stt, end);

	matrix SolAVX;
	stt = omp_get_wtime();
	SolAVX.LeastSquaresEstimation_Parallel_AVX(DesMat, ObsVec);    //OpenMP(AVX2)并行实现(在“OpenMP并行实现”的基础上引入向量化)
	end = omp_get_wtime();
	CoutResult(SolRef, SolAVX, stt, end);

	return 0;
}
