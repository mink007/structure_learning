#%matplotlib inline
#Using python 2.7*

#This file is for small.csv data set. Please change the inputfilename and outputfilename paths for medium and large data sets accordingly.
import sys

import networkx as nx
import pandas as pd
import numpy as np
import math
import time
import copy

import ntpath
import matplotlib
import matplotlib.pyplot as plt

import pdb
from collections import defaultdict 

start_time = time.time()

def write_gph(good_G, filename):
    with open(filename, 'w') as f:
        
        #Writing all the edges present
        for edge in good_G.edges():
            f.write("{}, {}\n".format(edge[0], edge[1]))
    f.close()
    

    #plot_g.clear()
    plot_g = nx.DiGraph()
    list_node = []
    for node in good_G.nodes():
        list_node.append(node)
    #list_node.unique()
    #Remove duplicate nodes if present to deepcopy issues if any
    list_node = list(dict.fromkeys(list_node))
    for node in list_node:
        plot_g.add_node(node)
        for ed in good_G.edges():
            plot_g.add_edge(ed[0],ed[1])
            
    
    #posi = nx.spring_layout(plot_g,k=0.15,iterations=20)
    #For drawing the graph
    df = pd.DataFrame(index=plot_g.nodes(), columns=plot_g.nodes())
    for row, data in nx.shortest_path_length(plot_g):
        for col, dist in data.items():
            df.loc[row,col] = dist
            
    df = df.fillna(df.max().max())
    posi = nx.kamada_kawai_layout(plot_g, dist=df.to_dict())
    
    nx.draw_networkx(plot_g, arrows=True, with_labels=True, width=2.0,edge_color='red',node_color='orange',pos=posi)
    g_file = filename + ".png"
    print("SHOWING GRAPH")
    plt.axis('off')
    plt.savefig(g_file, bbox_inches='tight')
    plt.show() 
    
    
    

def write_score_and_time(graph_score,time_diff,outfile):
    with open(outfile, 'w') as f:
            f.write("{}, {}\n".format(graph_score, time_diff))
    f.close()
    
def write_graph_score(good_G, outfile, graph_score, time_diff):
    write_gph(good_G, outfile)
    outfile = ntpath.basename(outfile)
    outfile = ntpath.splitext(outfile)[0] + ".score_and_time"
    write_score_and_time(graph_score,time_diff,outfile)
    


def compute(infile, outfile):
    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING
    data = pd.read_csv(infile)
    #print("COLS", data.columns[0])
    x_i = data.columns
    
    #Initialize a graph
    G = nx.DiGraph()
    
    #Add all the nodes to the graph
    for i in range(0,len(x_i)):
        G.add_node(x_i[i])
    
    
    #Find m_ijk for graph G
    #Check if G is cyclic
    G, graph_score = find_graph_score(data,G)
    print("Graph Score", graph_score)
    #pdb.set_trace()
    node_list = []
    for n in G.nodes():
        node_list.append(n)
        #print("N-Node", n,node_list)
    
    good_G=nx.DiGraph()
    good_G = copy.deepcopy(G)
    initial_G=initial_GP=nx.DiGraph()
    initial_G = copy.deepcopy(G)
    print("Time taken to get this best score %s seconds ---" % (time.time() - start_time))
    write_graph_score(G, outfile, graph_score, (time.time() - start_time))
    
    #top_graph_score = 0
    #Graph iterations and Bayesian score calculations.
    
    #The Local Directed Graph Search Loop
    for nn in range(0,len(node_list)+1):
        node_list.append(node_list.pop(0))
        print("NODE LIST", node_list)
        #The K2Search loop starts here
        for n in range(0,len(node_list)):
            initial_G = copy.deepcopy(good_G)
            for o in range(n+1,len(node_list)):
                initial_GP = copy.deepcopy(initial_G)
                #print("Adding edge",node_list[o],node_list[n])
                initial_G.add_edge(node_list[o],node_list[n])
                initial_G, new_graph_score = find_graph_score(data,initial_G)
                print("Score", new_graph_score)
                if new_graph_score == "":
                    #print("Removing edge a",node_list[o],node_list[n])
                    initial_G.clear()
                    initial_G = copy.deepcopy(initial_GP)
                    continue 
                if new_graph_score > graph_score:
                    #print("Better score %f than previous bettwe of %f" % new_graph_score % graph_score)
                    print("Better score {} obtained, than previous best of {}\n".format(new_graph_score, graph_score))
                    good_G.clear()
                    good_G = copy.deepcopy(initial_G)
                    graph_score = new_graph_score
                    print("Time taken to get this best score %s seconds ---" % (time.time() - start_time))
                    write_graph_score(good_G, outfile, graph_score, (time.time() - start_time))                    
                #print("Removing edge b",node_list[o],node_list[n])    
                initial_G.clear()
                initial_G = copy.deepcopy(initial_GP)
        
        
    
def find_graph_score(data,G):
    is_cyclic_graph = 0
    try:
        is_cyclic_graph = nx.find_cycle(G, orientation='original')
        is_cyclic_graph = 1
        #print("cyclic graph")
    except:
        is_cyclic_graph = 0
        #print("Not cyclic graph")
    
    if(is_cyclic_graph == 0):
        m_ijk, G = find_m_ijk(data, G)
        score = find_bayesian_score(G, m_ijk)
        #print("Score = ", score)
        return G, score
    else:
        return G, ""
        

def find_bayesian_score(G, m_ijk):
    i = 1
    a_ij0 = defaultdict(list)
    m_ij0 = defaultdict(list)
    score = 0
    for n in G.nodes:
        first_term = 0
        for j in range(0,len(G.nodes[n]["list_values_at_j"])):
            index_a = str(i) + str(j+1) + "0"
            a_ij0[index_a] = 0
            m_ij0[index_a] = 0
            for k in range(0,len(G.nodes[n]["value_of_ri"])):
                index = str(i) + str(j+1) + str(k+1)
                a_ij0[index_a] = a_ij0[index_a] + 1 
                m_ij0[index_a] = m_ij0[index_a] + m_ijk[index]
                #print("INDEX_A", index_a)
            third_term = 0
            for k in range(0,len(G.nodes[n]["value_of_ri"])):
                index = str(i) + str(j+1) + str(k+1)
                #print("INDEX", index)
                third_term = third_term + math.lgamma(1 + m_ijk[index])
            first_term = first_term + math.lgamma(a_ij0[index_a]) - math.lgamma(a_ij0[index_a]+m_ij0[index_a]) + third_term
        i = i + 1
        score = score + first_term
    return score

def find_m_ijk(data, G):
    
    node_names_dict=defaultdict(list)
    a_count = 1
    min_val = defaultdict(list)
    max_val = defaultdict(list)
    total_count_values = defaultdict(list)
    for i in G.nodes:
            node_names_dict[a_count] = i
            a_count=a_count+1
            min_val[i]=np.min(data[i][:])
            max_val[i]=np.max(data[i][:])
            total_count_values[i] = max_val[i] - min_val[i] + 1
            G.nodes[i]["val_range_count"] = max_val[i] - min_val[i] + 1
            G.nodes[i]["range"] = range(min_val[i],max_val[i]+1, 1)
            G.nodes[i]["list_values_at_j"] = []
            G.nodes[i]["value_of_ri"] = []
            G.nodes[i]["parents"] = []
            

    for sam in range(0,data.shape[0]):
        for i in range(1,len(node_names_dict)+1):
            G.nodes[node_names_dict[i]]["data_value"]=data[node_names_dict[i]][sam]
            #G.nodes[node_names_dict[i]]["list_of_parents"] = []
            #G.nodes[node_names_dict[i]]["value_of_k"] = []
      
        for i in range(1,len(node_names_dict)+1):
            #G.nodes[node_names_dict[i]]["data_value"]=data[node_names_dict[i]][sam]
            list_of_parents = []
            for key in G.predecessors(node_names_dict[i]):
                list_of_parents.append(key)
            list_of_parents.sort()
                
            if len(list_of_parents)>0:
                G.nodes[node_names_dict[i]]["parents"] = list_of_parents
                #print("PARENTS", len(list_of_parents))
            #else:
                #print("No PARENTS")
                
            
            j_value_parent_combination = ""
            for parent_node in list_of_parents:
                j_value_parent_combination = j_value_parent_combination + str(G.nodes[parent_node]["data_value"])
            #print("Node J value parent combination", node_names_dict[i], j_value_parent_combination)
            #Store the parent node value combination and the index will give 'j'
            if len(list_of_parents)>0:
                if (str(j_value_parent_combination) not in G.nodes[node_names_dict[i]]["list_values_at_j"]):
                    G.nodes[node_names_dict[i]]["list_values_at_j"].append(j_value_parent_combination)
                    G.nodes[node_names_dict[i]]["value_of_ri"].append(G.nodes[node_names_dict[i]]["data_value"])
                    
                    
                    
            if len(list_of_parents) == 0:
                G.nodes[node_names_dict[i]]["list_values_at_j"] = [1]
                if (G.nodes[node_names_dict[i]]["data_value"] not in G.nodes[node_names_dict[i]]["value_of_ri"]):
                    G.nodes[node_names_dict[i]]["value_of_ri"].append(G.nodes[node_names_dict[i]]["data_value"])
            
    
    m_ijk = defaultdict(list)
    #for sam in range(0,data.shape[0]):
    #pdb.set_trace()
    for i in range(1,len(node_names_dict)+1):
        for j in range(0,len(G.nodes[node_names_dict[i]]["list_values_at_j"])):
            for k in range(0,len(G.nodes[node_names_dict[i]]["value_of_ri"])):
                #print("IJK", i, j+1 , k+1)
                index = str(i) + str(j+1) + str(k+1)
                m_ijk[index] = 0
                #pdb.set_trace()
                for sam in range(0,data.shape[0]):
                    j_parent_node_val = ""
                    for parents in G.nodes[node_names_dict[i]]["parents"]:
                        j_parent_node_val = j_parent_node_val + str(data[parents][sam])
                    
                    if j_parent_node_val == "":
                        j_parent_node_val = 1
                    
                    #print("j_parent_node val", j_parent_node_val )
                    #print("j parane node val from data", G.nodes[node_names_dict[i]]["list_values_at_j"][j] )
                    if j_parent_node_val == G.nodes[node_names_dict[i]]["list_values_at_j"][j]:
                        k_node_value = data[node_names_dict[i]][sam]
                        if k_node_value == G.nodes[node_names_dict[i]]["value_of_ri"][k]:
                            m_ijk[index] = m_ijk[index] + 1
    #print("MIJK", m_ijk)
    return m_ijk, G
    

def main():
    #if len(sys.argv) != 3:
    #    raise Exception("usage: python project1.py <infile>.csv <outfile>.gph"
    #inputfilename = sys.argv[1]
    #outputfilename = sys.argv[2]
    inputfilename = "C:/Users/mink_/AA228Student/workspace/project1/small.csv"
    outputfilename = "C:/Users/mink_/AA228Student/workspace/project1/small.gph"
    print("IN/OUT FILES", inputfilename, outputfilename)
    compute(inputfilename, outputfilename)
    
if __name__ == '__main__':
    main()


