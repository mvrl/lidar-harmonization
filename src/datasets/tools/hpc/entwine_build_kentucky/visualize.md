
ssh -L 8085:localhost:8085 lcc
conda activate entwine
http-server /pscratch/nja224_uksr/ky_lidar/entwine_all -p 8085 --cors

#ssh -L 8085:localhost:8085 lcc

http://potree.entwine.io/data/view.html?r=%22http://localhost:8085/%22

http://dev.speck.ly/?s=0&r=ept://localhost:8085/&c0s=local://color

