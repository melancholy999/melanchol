import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

# open('reward_log.txt', 'w').close()

plt.style.use('ggplot')
pause = False

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,2,1)
ax2 = fig1.add_subplot(1,2,2)
fig2 = plt.figure()
bx1 = fig2.add_subplot(1,2,1)
bx2 = fig2.add_subplot(1,2,2)

def animate(i):
    graph_data = open('reward_log.txt','r').read()
    lines = graph_data.split('\n')
    xs = []
    ys1 = []
    ys2 = []
    vcs = []
    pcs = []
    for line in lines:
        if len(line) > 1:
            x, y1 , y2, vc, pc = line.split(',')
            xs.append(float(x))
            ys1.append(float(y1))
            ys2.append(float(y2))
            vcs.append(float(vc))
            pcs.append(float(pc))
            
    ax1.clear()
    ax2.clear()
    bx1.clear()
    bx2.clear()
    ax1.plot(xs, ys1)
    ax2.plot(xs, ys2, 'm')
    bx1.plot(xs, vcs, 'g')
    bx2.plot(xs, pcs, 'b')
    ax1.set_xlabel('Episode No.')
    ax1.set_ylabel('Episode Reward')
    ax2.set_xlabel('Episode No.')
    ax2.set_ylabel('Avearge Episode Rewards')
    ax1.set_title('Reward Graph')
    ax2.set_title('Avg Reward over 100 episodes')
    bx1.set_xlabel('Episode No.')
    bx1.set_ylabel('Value Cost')
    bx2.set_xlabel('Episode No.')
    bx2.set_ylabel('Policy cost')
    bx1.set_title('Value function model')
    bx2.set_title('Policy Function Model')
    if len(xs) == 5000:
        fig1.savefig('Fig_1.png')
        fig2.savefig('Fig_2.png')
        quit()
    
def onClick(event):
    global pause
    pause ^= True
    
ani1 = animation.FuncAnimation(fig1, animate, interval=1000)
ani2 = animation.FuncAnimation(fig2, animate, interval=1000)
plt.show()
