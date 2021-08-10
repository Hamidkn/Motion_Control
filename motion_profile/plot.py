import matplotlib.pyplot as plt

def draw_plots(t,x1,x2,Ftotal,curr1,curr2):
        _ , ax = plt.subplots(figsize=(12, 6))

        ax.plot(t,x1)
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel('Position (m)', fontsize=16)
        plt.savefig('motion_profile/Images/position.png')
        plt.show()

        _ , ax = plt.subplots(figsize=(12, 6))
        ax.plot(x1,Ftotal)
        ax.set_xlabel('position (m)', fontsize=16)
        ax.set_ylabel('Force (N)', fontsize=16)
        plt.savefig('motion_profile/Images/Force.png')
        plt.show()

        _ , ax = plt.subplots(figsize=(12, 6))
        ax.plot(x1,curr1)
        ax.plot(x1,curr2)
        ax.set_xlabel('position (m)', fontsize=16)
        ax.set_ylabel('Current (A)', fontsize=16)
        plt.savefig('motion_profile/Images/current.png')
        plt.show()

        _ , ax = plt.subplots(figsize=(12, 6))
        ax.plot(t,x2)
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel('Velocity (m/s)', fontsize=16)
        plt.savefig('motion_profile/Images/velocity.png')
        plt.show()  