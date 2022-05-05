import argparse
from train import run
from plot import plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='model')
    parser.add_argument('--lr', type=float, help='learning_rate')
    parser.add_argument('--dropout', type=float, help='dropout')

    args = parser.parse_args()
    print(args.model,args.lr,args.dropout)

    #python main.py --model lenet --lr 0.01 --dropout 0.0
    # run("lenet", learning_rate=0.01, dropout=0.5)

    train_loss,train_acc, test_loss, test_acc = run(model=args.model, learning_rate=args.lr, dropout=args.dropout)

    print(train_loss,train_acc, test_loss, test_acc)

    #绘图
    # plot(train_loss,train_acc, test_loss, test_acc)