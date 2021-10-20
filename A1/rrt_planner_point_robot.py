import time
import random
import drawSample
import math
import sys
import imageToRects
import utils
import os

# display = drawSample.SelectRect(imfile=im2Small,keepcontrol=0,quitLabel="")
args = utils.get_args()
visualize = utils.get_args()
drawInterval = 100  # 10 is good for normal real-time drawing

prompt_before_next = 1  # ask before re-running sonce solved
SMALLSTEP = args.step_size  # what our "local planner" can handle.
map_size, obstacles = imageToRects.imageToRects(args.world)
# Note the obstacles are the two corner points of a rectangle
# Each obstacle is (x1,y1), (x2,y2), making for 4 points
XMAX = map_size[0]
YMAX = map_size[1]

vertices = [
    [args.start_pos_x, args.start_pos_y],
    [args.start_pos_x, args.start_pos_y + 10],
]

# goal/target
tx = args.target_pos_x
ty = args.target_pos_y

# start
sigmax_for_randgen = XMAX / 2.0
sigmay_for_randgen = YMAX / 2.0
nodes = 0
edges = 1


def redraw(canvas, G):
    canvas.clear()
    canvas.markit(tx, ty, r=SMALLSTEP)
    drawGraph(G, canvas)
    for o in obstacles:
        canvas.showRect(o, outline="blue", fill="blue")
    canvas.delete("debug")


def drawGraph(G, canvas):
    global vertices, nodes, edges
    if not visualize:
        return
    for i in G[edges]:
        canvas.polyline([vertices[i[0]], vertices[i[1]]])


def genPoint():
    if args.rrt_sampling_policy == "uniform":
        # Uniform distribution
        x = random.random() * XMAX
        y = random.random() * YMAX
    elif args.rrt_sampling_policy == "gaussian":
        # Gaussian with mean at the goal
        x = random.gauss(tx, sigmax_for_randgen)
        y = random.gauss(ty, sigmay_for_randgen)
    else:
        print("Not yet implemented")
        quit(1)

    bad = 1
    while bad:
        bad = 0
        if args.rrt_sampling_policy == "uniform":
            # Uniform distribution
            x = random.random() * XMAX
            y = random.random() * YMAX
        elif args.rrt_sampling_policy == "gaussian":
            # Gaussian with mean at the goal
            x = random.gauss(tx, sigmax_for_randgen)
            y = random.gauss(ty, sigmay_for_randgen)
        else:
            print("Not yet implemented")
            quit(1)
        # range check for gaussian
        if x < 0:
            bad = 1
        if y < 0:
            bad = 1
        if x > XMAX:
            bad = 1
        if y > YMAX:
            bad = 1
    return [x, y]


def returnParent(k, canvas, G):
    """ Return parent note for input node k. """
    for e in G[edges]:
        if e[1] == k:
            canvas.polyline([vertices[e[0]], vertices[e[1]]], style=3)
            return e[0]


def genvertex():
    vertices.append(genPoint())
    return len(vertices) - 1


def pointToVertex(p):
    vertices.append(p)
    return len(vertices) - 1


def pickvertex():
    return random.choice(range(len(vertices)))


def lineFromPoints(p1, p2):
    """Compute slope of line from two points."""
    dx = p1[0] - p2[0]
    w = (p1[1] - p2[1]) / dx
    b = p1[1] - w * p1[0]
    return w, b


def pointPointDistance(p1, p2):
    """ Compute Euclidean distance between two points in 2D. """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def closestPointToPoint(G, p2):
    """ Return index of closest vertex on the graph to generated point """
    shortest_distance = float("inf")
    index = -1
    for i, v in enumerate(G[nodes]):
        distance = pointPointDistance(vertices[v], p2)
        if distance < shortest_distance:
            index = i
            shortest_distance = distance

    # return index of the closest vertex on the graph
    return index


def lineHitsRect(p1, p2, r):
    # segment slope
    w, b = lineFromPoints(
        p1, p2
    )  # y = wx + b for x [p1[0], p2[0]] and y [p1[1], p2[1]]

    # see if line intersects with all 4 sides of rectangle (first check if line to line intersection, then check if intersection point within segment)
    x1_y = r[0], w * r[0] + b  # left edge using x1
    x2_y = r[2], w * r[2] + b  # right edge using x2
    y1_x = (r[1] - b) / w, r[1]  # bottom edge using y1
    y2_x = (r[3] - b) / w, r[3]  # top edge using y2

    # get the four corners of the rectangle
    A = r[0], r[1]  # upper left
    B = r[2], r[1]  # upper right
    C = r[2], r[3]  # right bottom
    D = r[0], r[3]  # left bottom

    for p in [x1_y, x2_y, y1_x, y2_x]:
        if point_within_range(p, p1, p2) and (
            point_within_range(p, A, B)
            or point_within_range(p, B, C)
            or point_within_range(p, C, D)
            or point_within_range(p, D, A)
        ):
            return True
    return False


def point_within_range(point, p1, p2):
    x_range = [p2[0], p1[0]] if p1[0] > p2[0] else [p1[0], p2[0]]
    y_range = [p2[1], p1[1]] if p1[1] > p2[1] else [p1[1], p2[1]]
    return x_range[0] <= point[0] <= x_range[1] and y_range[0] <= point[1] <= y_range[1]


def inRect(p, rect, dilation):
    """ Return 1 in p is inside rect, dilated by dilation (for edge cases). """
    return (
        rect[0] - dilation <= p[0] <= rect[2] + dilation
        and rect[1] - dilation <= p[1] <= rect[-1] + dilation
    )


def steer(closest_point, x_rand):
    dx = x_rand[0] - closest_point[0]
    dy = x_rand[1] - closest_point[1]
    theta = math.atan2(dy, dx)

    y = math.sin(theta) * args.step_size
    x = math.cos(theta) * args.step_size
    return [closest_point[0] + x, closest_point[1] + y]


def rrt_search(G, tx, ty, canvas):
    global sigmax_for_randgen, sigmay_for_randgen
    n = 0
    nsteps = 0
    while 1:
        nsteps += 1
        # 1. sample random point in free space
        p = genPoint()

        # 2. find nearest point on the graph to sampled point
        v = closestPointToPoint(G, p)

        if visualize:
            # if nsteps%500 == 0: redraw()  # erase generated points now and then or it gets too cluttered
            n = n + 1
            if n > 10:
                canvas.events()
                n = 0

        # compute x_new in the direction of tthe generated point
        p = steer(vertices[v], x_rand=p)

        hit_obstacle = False
        for o in obstacles:
            # print(o)
            # canvas.showRect(o, outline='red', fill='red')
            # canvas.events()
            if lineHitsRect(vertices[v], p, o) or inRect(p, o, 1):
                hit_obstacle = True
                break
            # canvas.showRect(o, outline='red', fill='blue')

        # if the line between the sampled point and the closest point on the graph hits an obstacle, sample new point.
        if hit_obstacle:
            continue

        # 3. take a step in the direction of the sampled point if obstacle free
        k = pointToVertex(p)  # is the new vertex ID
        G[nodes].append(k)
        G[edges].append((v, k))
        if visualize:
            canvas.polyline([vertices[v], vertices[k]])

        if pointPointDistance(p, [tx, ty]) < SMALLSTEP:
            print("Target achieved.", len(vertices), "nodes in entire tree")
            print(f"RRT has ran for {nsteps} steps.")
            if visualize:
                t = pointToVertex([tx, ty])  # is the new vertex ID
                G[edges].append((k, t))
                if visualize:
                    canvas.polyline([p, vertices[t]], 1)
                # while 1:
                #     # backtrace and show the solution ...
                #     canvas.events()
                rrt_nsteps = nsteps
                nsteps = 0
                totaldist = 0
                while 1:
                    oldp = vertices[k]  # remember point to compute distance
                    k = returnParent(k, canvas, G)  # follow links back to root.
                    canvas.events()
                    if k <= 1:
                        break  # have we arrived?
                    nsteps = nsteps + 1  # count steps
                    totaldist = totaldist + pointPointDistance(
                        vertices[k], oldp
                    )  # sum lengths
                print("Path length", totaldist, "using", nsteps, "nodes.")
                # save canvas
                import pyscreenshot as ImageGrab
                x0 = canvas.canvas.winfo_rootx()
                y0 = canvas.canvas.winfo_rooty()
                x1 = x0 + canvas.canvas.winfo_width()
                y1 = y0 + canvas.canvas.winfo_height()

                im = ImageGrab.grab(bbox=(x0,y0,x1,y1))
                im.save(os.path.join("graph",f'stepsize-{args.step_size}-seed-{args.seed}.png'))
                # writing stats
                with open("point_robot_logs", "a") as f:
                    f.write(f"{args.seed},{args.step_size},{rrt_nsteps},{totaldist},{nsteps},{len(vertices)}\n")
                exit(0)

def main():
    # seed
    random.seed(args.seed)
    if visualize:
        canvas = drawSample.SelectRect(
            xmin=0, ymin=0, xmax=XMAX, ymax=YMAX, nrects=0, keepcontrol=0
        )  # , rescale=800/1800.)
        for o in obstacles:
            canvas.showRect(o, outline="red", fill="blue")
    while 1:
        # graph G
        G = [[0], []]  # nodes, edges
        redraw(canvas, G)
        G[edges].append((0, 1))
        G[nodes].append(1)
        if visualize:
            canvas.markit(tx, ty, r=SMALLSTEP)

        drawGraph(G, canvas)
        rrt_search(G, tx, ty, canvas)

    if visualize:
        canvas.mainloop()


if __name__ == "__main__":
    main()
