def th(prms: str):
    depth_maps_d = prms["depth_maps_dir"]
    output_me = prms["output_method"]
    print(depth_maps_d)
    print(output_me)

prameters = {
    "depth_maps_dir": "depth_maps_dir666",
    "output_method": "output_method666"
}

th(prameters)