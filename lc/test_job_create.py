import dsr.job

h = dsr.job.host_configure()

config = h("my_job","my_account",projects_dir="striped/projects/dsr")

print("**********************************************************************************************************************")
print("PROJECT CONFIGURATION PARAMETERS")
print("**********************************************************************************************************************")


for i,c in enumerate(config):
    print("({}) {} : {}".format(i,c,config[c]))

c = dsr.job.create_project(config)
c()