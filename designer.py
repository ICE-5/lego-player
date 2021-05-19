import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import csv, copy
import keyboard
import networkx as nx

class Area():
    def __init__(self, tx, ty, bx, by) -> None:
        self.area = [(tx, ty), (bx, by)]
        self.top_left = (tx, ty)
        self.bottom_right = (bx, by)
        self.tx = tx
        self.ty = ty
        self.bx = bx
        self.by = by


class Brick():
    def __init__(self, anchor:np.array, extents:List[float], rotation:List[bool]=[0, 0, 0]) -> None:
        self.extents = extents
        self.anchor = anchor
        self.sides = extents.copy()
        self.rotate(rotation)
    

    def rotate(self, rotation:List[bool]=[0, 0, 0]) -> None:
        # apply rotation
        self.rotation = rotation
        if rotation[0]:
            self.sides = [self.sides[0], self.sides[2], self.sides[1]]
        if rotation[1]:
            self.sides = [self.sides[2], self.sides[1], self.sides[0]]
        if rotation[2]:
            self.sides = [self.sides[1], self.sides[0], self.sides[2]]
        self.get_bounding()
    
    def move(self, new_anchor:np.array) -> None:
        # move to new anchor point
        self.anchor = new_anchor
        self.get_bounding()


    def get_bounding(self) -> None:
        # set anchor as center of brick
        # clockwise
        a, b, c = [i / 2 for i in self.sides]
        self.center = self.anchor
        self.lower_NW = self.center + np.array([-a, -b, -c])
        self.lower_SE = self.center + np.array([ a,  b, -c])
        self.lower_NE = self.center + np.array([ a, -b, -c])
        self.lower_SW = self.center + np.array([-a,  b, -c])
        self.upper_NW = self.center + np.array([-a, -b,  c])
        self.upper_NE = self.center + np.array([ a, -b,  c])
        self.upper_SE = self.center + np.array([ a,  b,  c])
        self.upper_SW = self.center + np.array([-a,  b,  c])
        self.bounding = np.concatenate([self.lower_NW, 
                                        self.lower_NE,
                                        self.lower_SE,
                                        self.lower_SW,
                                        self.upper_NW,
                                        self.upper_NE,
                                        self.upper_SE,
                                        self.upper_SW])
        # general features
        self.lower_level = round(self.lower_NW[2] / self.extents[2])
        self.upper_level = round(self.upper_NW[2] / self.extents[2])
        self.lower_z = self.lower_NW[2]
        self.height = round(self.sides[2] / self.extents[2])
        
        

class Designer():
    def __init__(self, arena_length:float, brick_extents:List[float]) -> None:
        self.built = []
        self.brick_extents = brick_extents
        self.arena_length = arena_length
        self.mod_x, self.mod_y, self.mod_z = brick_extents

        self.max_level = 0

        self.occupy_registry = {}
        self.base_registry = {}
        self.base_registry[0] = [(-1, Area(0., 0., arena_length, arena_length)), ]

        # graph
        self.G = nx.DiGraph()


    
    def __len__(self) -> int:
        return len(self.built)
    
    def __getitem__(self, idx: int) -> Brick:
        return self.built[idx]

    def next(self, num_trial:int=10, level_range:int=3, logfile:str=None, logmode:str="a") -> Optional[Tuple[Brick, Area]]:
        # try on each level
        for l in range(max(self.max_level-level_range, 0), self.max_level+level_range):
            for _ in range(num_trial):                                          
                rotation = self.rotate_by_probability()
                brick = Brick(np.zeros(3), self.brick_extents, rotation)
                try:
                    anchor, bgr, base_brick_idx_list = self._search_valid_position_level(l, brick, num_trial)
                    brick.move(anchor)
                    self.update_registry(brick, base_brick_idx_list)

                    # logging
                    if logfile is not None:
                        with open(logfile, logmode, newline='') as x:
                            writer = csv.writer(x, delimiter=",")
                            writer.writerow(list(brick.lower_NW))
                            writer.writerow(list(brick.upper_SE))
                            writer.writerow([bgr.tx, bgr.ty, brick.lower_z])
                            writer.writerow([bgr.bx, bgr.by, brick.lower_z])
                        x.close()
                        

                    return brick
                except:
                    pass
        return None

    
    def generate_design(self, num_trial:int=10, num_brick:int=10, logfile:str="log.csv") -> None:
        for i in range(num_brick):
            logmode = "w" if i == 0 else "a"
            self.next(num_trial=num_trial, logfile=logfile, logmode=logmode)
        
    
    def update_registry(self, brick:Brick, base_brick_idx_list:List[int]) -> None:
        # add to occupy registry
        brick_idx = len(self.built)
        area = Area(brick.lower_NW[0], brick.lower_NW[1], brick.lower_SE[0], brick.lower_SE[1])

        for l in range(brick.lower_level, brick.upper_level):
            try:
                self.occupy_registry[l].append(area)
            except:
                self.occupy_registry[l] = [area, ]

        # add to base registry
        try:
            self.base_registry[brick.upper_level].append((brick_idx, area))
        except:
            self.base_registry[brick.upper_level] = [(brick_idx, area),]

        # update max level reference
        self.max_level = max(self.occupy_registry.keys())

        # add to built
        self.built.append(brick)

        # add node to graph
        self.G.add_node(brick_idx)
        # add directed edge (from base to upper) to graph
        if len(base_brick_idx_list) != 0:
            for i in base_brick_idx_list:
                self.G.add_edge(i, brick_idx)
        
        print(brick_idx, base_brick_idx_list)
    

    def _search_valid_position_level(self, level:int, brick:Brick, num_trial:int) -> Optional[Tuple[np.array, Area]]:
        # get occupy constraints
        occupied_area_list = []
        for l in range(level, level + brick.height):
            try:
                occupied_area_list.extend(self.occupy_registry[l])
            except:
                pass
        try:
            occupied_gr_list = [self._get_golden_rectangle(oa, brick.sides) for oa in occupied_area_list]
        except:
            occupied_gr_list = None

        # get base constraints
        try:
            base_areatuple_list = self.base_registry[level]
            base_gr_list = [(i, self._get_golden_rectangle(ba, brick.sides)) for i, ba in base_areatuple_list]
        except:
            return None                 # if no base area for placement, then the search fails

        for _ in range(num_trial):
            # randomly choose a valid base area (ba) for placement 
            num_base_area = len(base_gr_list)
            bgr = base_gr_list[np.random.choice(num_base_area)][1]

            if level == 0:
                bgr = base_gr_list[0][1]
            x = round(np.random.random_sample() * (bgr.bx - bgr.tx) + bgr.tx, 2)
            y = round(np.random.random_sample() * (bgr.by - bgr.ty) + bgr.ty, 2)

            collided = False
            if occupied_gr_list is None:
                pass
            else:
                for ogr in occupied_gr_list:
                    if self._check_collided(ogr, brick.sides, (x, y)):
                        collided = True

            if not collided:
                lower_NW = np.array([x, y, level * self.mod_z])
                anchor = lower_NW + np.array([i/2 for i in brick.sides])

                # find indices of touched base bricks
                base_brick_idx_list = []
                for idx, bgr in base_gr_list:
                    if self._check_collided(bgr, brick.sides, (x, y)) and idx != -1:
                        base_brick_idx_list.append(idx)

                return (anchor, bgr, base_brick_idx_list)

        return None


    def _get_golden_rectangle(self, base:Area, sides:List[float]) -> Area:
        a, b, _ = sides
        gr = Area(max(base.tx - a, 0.), 
                  max(base.ty - b, 0.), 
                  min(base.bx, self.arena_length - a),
                  min(base.by, self.arena_length - b)
                  )
        return gr

        
    def _check_collided(self, base:Area, sides:List[float], coordinates:List[float]) -> bool:
        x, y = coordinates
        gr = self._get_golden_rectangle(base, sides)

        if (x >= gr.tx and x <= gr.bx) and (y >= gr.ty and y <= gr.by):
            return True
        else:
            return False


    @staticmethod
    def rotate_by_probability(prob:List[float]=[0.1, 0.1, 0.5]) -> List[float]:
        return [np.random.binomial(1, prob[0]), np.random.binomial(1, prob[1]), np.random.binomial(1, prob[2])]

    

if __name__ == "__main__":
    NUM_BRICK=30
    d = Designer(arena_length=1., brick_extents=[0.4, 0.2, 0.1])
    d.generate_design(num_brick=NUM_BRICK)

    # visualize graph
    node_color = []
    for node in d.G.nodes:
        degree = d.G.degree(node)

        if degree == 0:
            color = '#fcb045'
        elif degree == 1:
            color = '#fd7435'
        elif degree == 2:
            color = '#fd4c2a'
        elif degree == 3:
            color = '#fd1d1d'
        elif degree == 4:
            color = '#c22b65'
        else:
            color = '#833ab4'
    
        node_color.append(copy.deepcopy(color))

    nx.draw(d.G, 
            with_labels=True,
            node_color=node_color,
            node_size=200,
            alpha=0.9,
            edge_color='#575757',
            linewidths=None,
            font_weight='bold',
            font_size=8)
    plt.show()



    ## debug and step 3D visualization in Rhino only
    # logfile = "debug.csv"
    # for i in range(NUM_BRICK):
    #     logmode = "w" if i==0 else "a"

    #     keyboard.wait("n")
    #     d.next(logfile=logfile, logmode=logmode)


    
            
               
