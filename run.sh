for i in {1..6}
do
    echo "Training Model for scenario $i"
    echo "SIZE 4096"
    python3 train.py --data_folder ../datasets/data/ --results_folder ../results/ --scenario $i --size 4096
    echo ""
    echo "SIZE 512"
    python3 train.py --data_folder ../datasets/data/ --results_folder ../results/ --scenario $i --size 512
    echo -e "\n"
done
