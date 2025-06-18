% MatDat：矩阵数据
% MatNam：文件名称
% WrtMod：行列模式（false - 行优先、true - 列优先）
function WriteBinMat_SGF(MatDat,MatNam,WrtMod)
    if ~isnumeric(MatDat)
        error("输入矩阵应为数值型！");
    end
    if ~islogical(WrtMod)
        error("行列模式应为逻辑型！");
    end
    if ~(ischar(MatNam) || isstring(MatNam))
        error("文件名称应为字符串！");
    end

    [Hgt,Wid] = size(MatDat);
    SingleFloat = single(MatDat);

    fid = fopen(MatNam, "w", "l");
    if fid == -1
        error("文件打开失败：%s！",MatNam);
    end
    fwrite(fid, 16,"uint32");   % 数据头的所占字节数
    fwrite(fid,  4,"uint32");   % “4”代表单精度浮点型数据
    fwrite(fid,Hgt,"uint32");   % 矩阵的行数
    fwrite(fid,Wid,"uint32");   % 矩阵的列数
    if ~WrtMod
        fwrite(fid,SingleFloat.',"single");
    else
        fwrite(fid,SingleFloat,"single");
    end
    fclose(fid);
end

