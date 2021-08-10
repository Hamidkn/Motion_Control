import matplotlib.pyplot as plt

def draw_plots(t,x1,x2,Ftotal,curr1,curr2):
        plt.plot(t,x1)
        plt.savefig('model_five_train/Images/position.png')
        plt.show()

        plt.plot(x1,Ftotal)
        plt.savefig('model_five_train/Images/Force.png')
        plt.show()

        plt.plot(x1,curr1)
        plt.plot(x1,curr2)
        plt.savefig('model_five_train/Images/current.png')
        plt.show()

        plt.plot(t,x2)
        plt.savefig('model_five_train/Images/velocity.png')
        plt.show()  