conda activate dualdn && cd /home/zhenying/qhong/repo/Restormer && python demo.py \
--task Real_Denoising \
--input_dir /home/zhenying/qhong/data/ssd2/from_xtchen/source \
--result_dir /home/zhenying/qhong/data/ssd2/from_xtchen/denoising \
&&
conda activate dualdn && cd /home/zhenying/qhong/repo/Flow-Anything && python infer.py \
--input /home/zhenying/qhong/data/ssd2/from_xtchen/denoising \
--out /home/zhenying/qhong/data/ssd2/from_xtchen/denoising_mma \
--cfg config/eval/spring-M.json \
--model /home/zhenying/qhong/repo/Flow-Anything/result/0302_fg_1500k/checkpoints/319k.pth \
&&
conda activate mm && cd /home/zhenying/qhong/repo/mm && python start.py config

