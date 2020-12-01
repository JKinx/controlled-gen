from controlled_gen import ControlledGen
import argparse
    
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, required=True, help='Model Name', choices=["dateSet"])
parser.add_argument('--api', type=str, required=True, help='Api Name', choices=["get_yz", "get_yz_templated",  "get_z"])
parser.add_argument('--template_id', type=int, help='id of template', default=-1, choices=[0,1,2,3,4,5,6,7])
parser.add_argument('--device', default="cuda", help='device for model')
args = parser.parse_args()

# make sure template is given
if args.api == "get_yz_templated":
    assert args.template_id != -1
    
# load model
model = ControlledGen(model_name = args.model, 
                      device=args.device)

x = (13, 6, 2020)
template_list = [[7,9,0,-1,2,-1,3,-1],
                 [9,9,0,-1,2,-1,9,-1,3,-1],
                 [7,9,2,-1,0,-1,4,-1,3,-1],
                 [9,9,2,-1,0,-1,9,-1,3,-1],
                 [7,9,8,1,-1,2,-1,4,-1,3,-1],
                 [9,9,8,1,-1,2,-1,9,-1,3,-1],
                 [7,9,2,-1,1,-1,4,-1,3,-1],
                 [9,9,2,-1,1,-1,9,-1,3,-1],
                ]
y = ['the', 'date', 'is', 'the', 'thirteen', 'of', 'june', ',', '2020', '.']

print("Testing " + args.api + " for " + args.model)
print('--'*30)

if args.api == "get_yz":
    out = model.get_yz(x)
    print("x: " +  " ".join([str(el) for el in x]))
    print()
    print("y: " + " ".join(out["y"]))
    print()
    print("z: " + " ".join([str(el) for el in out["z"]]))
elif args.api == "get_yz_templated":
    template = template_list[args.template_id]
    out = model.get_yz(x, template)
    print("x: " +  " ".join([str(el) for el in x]))
    print()
    print("y: " + " ".join(out["y"]))
    print()
    print("z template: " + " ".join([str(el) for el in template]))
    print()
    print("z: " + " ".join([str(el) for el in out["z"]]))   
elif args.api == "get_z":
    z = model.get_z(x, y)
    print("x: " +  " ".join([str(el) for el in x]))
    print()
    print("y: " + " ".join(y))
    print()
    print("z: " + " ".join([str(el) for el in z]))  
else:
    raise ValueError("Should not reach here")
