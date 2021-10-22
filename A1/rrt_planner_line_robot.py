import time
import random
import drawSample
import math
import sys
import imageToRects
import utils
import os

#display = drawSample.SelectRect(imfile=im2Small,keepcontrol=0,quitLabel="")
args = utils.get_args()
visualize = utils.get_args()
drawInterval = 100 # 10 is good for normal real-time drawing

prompt_before_next=1  # ask before re-running sonce solved
SMALLSTEP = args.step_size # what our "local planner" can handle.
map_size,obstacles = imageToRects.imageToRects(args.world)
#Note the obstacles are the two corner points of a rectangle
#Each obstacle is (x1,y1), (x2,y2), making for 4 points
XMAX = map_size[0]
YMAX = map_size[1]

vertices = [ [args.start_pos_x, args.start_pos_y], [args.start_pos_x, args.start_pos_y + 10]]

# goal/target
tx = args.target_pos_x
ty = args.target_pos_y

# start
sigmax_for_randgen = XMAX/2.0
sigmay_for_randgen = YMAX/2.0
nodes=0
edges=1

def redraw(canvas, G):
    canvas.clear()
    canvas.markit( tx, ty, r=SMALLSTEP )
    drawGraph(G, canvas)
    for o in obstacles: canvas.showRect(o, outline='blue', fill='blue')
    canvas.delete("debug")


def drawGraph(G, canvas):
    global vertices,nodes,edges
    if not visualize: return
    for i in G[edges]:
       canvas.polyline([vertices[i[0]], vertices[i[1]]])


def genPoint():
    if args.rrt_sampling_policy == "uniform":
        # Uniform distribution
        x = random.random()*XMAX
        y = random.random()*YMAX
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
            x = random.random()*XMAX
            y = random.random()*YMAX
        elif args.rrt_sampling_policy == "gaussian":
            # Gaussian with mean at the goal
            x = random.gauss(tx, sigmax_for_randgen)
            y = random.gauss(ty, sigmay_for_randgen)
        else:
            print("Not yet implemented")
            quit(1)
        # range check for gaussian
        if x<0: bad = 1
        if y<0: bad = 1
        if x>XMAX: bad = 1
        if y>YMAX: bad = 1
    return [x,y]

def returnParent(k, canvas, G):
    """ Return parent note for input node k. """
    for e in G[edges]:
        if e[1]==k:
            canvas.polyline([vertices[e[0]], vertices[e[1]]], style=3)
            return e[0]

def genvertex():
    vertices.append( genPoint() )
    return len(vertices)-1

def pointToVertex(p):
    vertices.append( p )
    return len(vertices)-1

def pickvertex():
    return random.choice( range(len(vertices) ))

def lineFromPoints(p1,p2):
    """Compute slope of line from two points."""
    dx = p1[0] - p2[0]
    if dx == 0:
        raise Exception("Vertical line.")
    w = (p1[1] - p2[1]) / dx
    b = p1[1] - w * p1[0]
    return w, b

def pointPointDistance(p1,p2):
    """ Compute Euclidean distance between two points in 2D. """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def closestPointToPoint(G,p2):
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

def lineHitsRect(p1,p2,r):
    dx = p1[0] - p2[0] 
    if dx == 0:
        if r[0] <= p1[0] <= r[2]:
            largest_point = (p2[1],p1[1]) if p1[1] > p2[1] else (p1[1], p2[1])
            if r[1] > largest_point[1] or largest_point[0] > r[3]:
                return False
            else:
                return True
        else:
            return False

    # segment slope
    w, b = lineFromPoints(
        p1, p2
    )  # y = wx + b for x [p1[0], p2[0]] and y [p1[1], p2[1]]

    # if we have a horizontal line
    if w == 0:
        if r[1] <= b <= r[3]:
            largest_point = (p2[0],p1[0]) if p1[0] > p2[0] else (p1[0], p2[0])
            if r[0] > largest_point[1] or largest_point[0] > r[2]:
                return False
            else:
                return True
        else:
            return False
        

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

def point_within_canvas(p): 
    return 0 <= p[0] <= XMAX and 0 <= p[1] <= YMAX

def inRect(p,rect,dilation):
    """ Return 1 in p is inside rect, dilated by dilation (for edge cases). """
    return (
        rect[0] - dilation <= p[0] <= rect[2] + dilation
        and rect[1] - dilation <= p[1] <= rect[-1] + dilation
    )

def steer(closest_point, x_rand):
    if pointPointDistance(closest_point, x_rand) > SMALLSTEP:
        dx = x_rand[0] - closest_point[0]
        dy = x_rand[1] - closest_point[1]
        theta = math.atan2(dy, dx)

        y = math.sin(theta) * args.step_size
        x = math.cos(theta) * args.step_size
        return [closest_point[0] + x, closest_point[1] + y]
    else: 
        return x_rand

def can_turn(current_pos, current_orientation, target_orientation, obstacles, canvas):
    """ See if the robot can rotate w.r.t the center of the line robot to the target orientation. """

    # rad orientation to degrees and get the positive angle
    target_orientation_deg = 360 + round(math.degrees(target_orientation)) if round(math.degrees(target_orientation)) < 0 else round(math.degrees(target_orientation))
    current_orientation_deg = 360 + round(math.degrees(current_orientation)) if round(math.degrees(current_orientation)) < 0 else round(math.degrees(current_orientation))

    turn_left, turn_right = True, True
    
    padded = False
    if target_orientation_deg < current_orientation_deg: 
        padded = True
        target_orientation_deg += 360
    
    # if current_orientation_deg > target_orientation_deg:
    for angle in range(current_orientation_deg, target_orientation_deg + 1):
        angle = angle % 360
        robot_head, robot_tail = get_robot_head_tail(current_pos, math.radians(angle))
        if not point_within_canvas(robot_head) or not point_within_canvas(robot_tail): break
        for o in obstacles:
            if lineHitsRect(robot_head, robot_tail, o):
                canvas.canvas.create_line([robot_head, robot_tail],fill='yellow', width=3)
                canvas.events()
                turn_left = False
                break
        if not turn_left:
            break
        else:
            return True

    for angle in range(current_orientation_deg if padded else target_orientation_deg, target_orientation_deg - 361 if padded else current_orientation_deg - 1, -1):
        robot_head, robot_tail = get_robot_head_tail(current_pos, math.radians(angle))
        if not point_within_canvas(robot_head) or not point_within_canvas(robot_tail): break
        for o in obstacles:
            if lineHitsRect(robot_head, robot_tail, o):
                canvas.canvas.create_line([robot_head, robot_tail],fill='yellow', width=3)
                canvas.events()
                turn_right = False
                break
        if not turn_right:
            break
        else:
            return True
    return False

def get_robot_head_tail(position, orientation):
    dx = math.cos(orientation) * (args.robot_length/2)
    dy = math.sin(orientation) * (args.robot_length/2)
    robot_head = position[0] + dx, position[1] + dy
    robot_tail = position[0] - dx, position[1] - dy
    return robot_head, robot_tail

def get_turn_angle(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    theta_rad = math.atan2(dy, dx)
    return theta_rad

def rrt_search(G, tx, ty, canvas):
    #TODO
    #Fill this function as needed to work ...


    global sigmax_for_randgen, sigmay_for_randgen
    n=0
    nsteps=0
    current_orientation = args.start_pos_theta
    while 1:
        nsteps += 1
        p = genPoint()
        v = closestPointToPoint(G,p)

        if visualize:
            # if nsteps%500 == 0: redraw()  # erase generated points now and then or it gets too cluttered
            n=n+1
            if n>10:
                canvas.events()
                n=0

        nearest_vertex = vertices[v]

        # compute x_new in the direction of tthe generated point
        p = steer(nearest_vertex, x_rand=p)
        
        # NOTE: we assume that (x,y) represent the middle of the line robot and that theta is expressed w.r.t the x-axis
        # NOTE: we assume that x_new's orientation will be defined by the angle between the nearest neighbor and itself.
        theta = get_turn_angle(p,nearest_vertex)
        robot_head, robot_tail = get_robot_head_tail(p, theta)
        
        hit_obstacle = True
        
        for o in obstacles:
            if inRect(p,o,1) or inRect(robot_tail,o,1) or inRect(robot_head,o,1) \
                or lineHitsRect(nearest_vertex,p,o) or lineHitsRect(robot_head, robot_tail, o) \
                or not point_within_canvas(robot_head) or not point_within_canvas(robot_tail):
                break
        else:
            hit_obstacle = False

        # can't turn towards target point or trajectory or resulting position hit obstacle
        if hit_obstacle or not can_turn(current_pos=nearest_vertex, current_orientation=current_orientation, obstacles=obstacles, target_orientation=theta, canvas=canvas):
            continue
        current_orientation = theta
        # visualize line robot
        canvas.canvas.create_line([robot_head, robot_tail],fill='green', width=1)
        k = pointToVertex( p )   # is the new vertex ID
        G[nodes].append(k)
        G[edges].append( (v,k) )
        if visualize:
            canvas.polyline(  [nearest_vertex, vertices[k] ]  )

        if pointPointDistance( p, [tx,ty] ) < SMALLSTEP:
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
                    if k <= 1: break  # have we arrived?
                    nsteps = nsteps + 1  # count steps
                    totaldist = totaldist + pointPointDistance(vertices[k], oldp)  # sum lengths
                print("Path length", totaldist, "using", nsteps, "nodes.")
                
                # save canvas
                import pyscreenshot as ImageGrab
                x0 = canvas.canvas.winfo_rootx()
                y0 = canvas.canvas.winfo_rooty()
                x1 = x0 + canvas.canvas.winfo_width()
                y1 = y0 + canvas.canvas.winfo_height()
                os.makedirs(os.path.join("graph", "line"), exist_ok=True)
                im = ImageGrab.grab(bbox=(x0,y0,x1,y1))
                im.save(os.path.join("graph", "line",f'robot_length-{args.robot_length}-seed-{args.seed}.png'))
                # writing stats
                with open("line_robot_logs", "a") as f:
                    f.write(f"{args.seed},{args.step_size},{rrt_nsteps},{args.robot_length},{totaldist},{nsteps},{len(vertices)}\n")
                exit(0)

def main():
    #seed
    random.seed(args.seed)
    if visualize:
        canvas = drawSample.SelectRect(xmin=0,ymin=0,xmax=XMAX ,ymax=YMAX, nrects=0, keepcontrol=0)#, rescale=800/1800.)
        for o in obstacles: canvas.showRect(o, outline='red', fill='blue')
    while 1:
        # graph G
        G = [[0],[]]   # nodes, edges
        redraw(canvas, G)
        G[edges].append((0,1))
        G[nodes].append(1)
        if visualize: canvas.markit( tx, ty, r=SMALLSTEP )

        drawGraph(G, canvas)
        rrt_search(G, tx, ty, canvas)

    if visualize:
        canvas.mainloop()

if __name__ == '__main__':
    main()