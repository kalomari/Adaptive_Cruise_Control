# adaptive cruise control using Fuzzy logic approach
# This version is used to plot the the membership functions and
# the defuzzification



#!/usr/bin/env python
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Generate universe variables
#   * velocity on subjective ranges [-3.5, 2.0]
#   * distance on subjective ranges [-80, 100]
#   * acceleration has a range of [-1, 1] in units of percentage points
x_vel = np.arange(-95, 95, 0.1)
x_dis = np.arange(-300, 101, 1)
x_acc  = np.arange(-0.5, 0.5, 0.01)

vel_fast = fuzz.trimf(x_vel, [-95, -95, 0])
vel_ok = fuzz.trimf(x_vel, [-95, 0, 95])
vel_slow = fuzz.trimf(x_vel, [0, 95, 95])
dis_far = fuzz.trimf(x_dis, [-300, -300, 0])
dis_ok = fuzz.trimf(x_dis, [-300, 0, 100])
dis_close = fuzz.trimf(x_dis, [0, 100, 100])
decelerate = fuzz.trimf(x_acc, [-0.5, -0.5, 0])
const = fuzz.trimf(x_acc, [-0.5, 0, 0.5])
accelerate = fuzz.trimf(x_acc, [0, 0.5, 0.5])


# Visualize these universes and membership functions
fig, (ax1, ax0, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_vel, vel_slow, 'b', linewidth=1.5, label='slow')
ax0.plot(x_vel, vel_ok, 'g', linewidth=1.5, label='ok')
ax0.plot(x_vel, vel_fast, 'r', linewidth=1.5, label='fast')
#ax0.set_title('Car speed')
ax0.set_xlabel('Delta speed [cm/s]')
ax0.set_ylabel('Degree of membership')
ax0.legend()

ax1.plot(x_dis, dis_close, 'r', linewidth=1.5, label='Close')
ax1.plot(x_dis, dis_ok, 'g', linewidth=1.5, label='ok')
ax1.plot(x_dis, dis_far, 'b', linewidth=1.5, label='Far')
#ax1.set_title('Car distance')
ax1.set_xlabel('Delta Distance [cm]')
ax1.set_ylabel('Degree of membership')
ax1.legend()

ax2.plot(x_acc, decelerate, 'b', linewidth=1.5, label='decelerate')
ax2.plot(x_acc, const, 'g', linewidth=1.5, label='const')
ax2.plot(x_acc, accelerate, 'r', linewidth=1.5, label='accelerate')
#ax2.set_title('accleration')
ax2.set_xlabel('Delta Acceleration [cm/s' + u"\u00B2" + ']')
ax2.set_ylabel('Degree of membership')
ax2.legend()

# Turn off top/right axes
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.show()

####################################################################

# We need the activation of our fuzzy membership functions at these values.
# The exact values 6.5 and 9.8 do not exist on our universes...
# This is what fuzz.interp_membership exists for!
vel= 50
dis= -170

vel_level_slow = fuzz.interp_membership(x_vel, vel_slow, vel)
vel_level_ok= fuzz.interp_membership(x_vel, vel_ok, vel)
vel_level_fast = fuzz.interp_membership(x_vel, vel_fast, vel)

dis_level_close = fuzz.interp_membership(x_dis, dis_close, dis)
dis_level_ok= fuzz.interp_membership(x_dis, dis_ok, dis)
dis_level_far = fuzz.interp_membership(x_dis, dis_far, dis)

print(vel_level_slow, vel_level_ok, vel_level_fast)
print(dis_level_close, dis_level_ok, dis_level_far)

rule1= np.fmin(dis_level_close, vel_level_slow)
rule2= np.fmin(dis_level_close, vel_level_ok)
rule3= np.fmin(dis_level_close, vel_level_fast)
rule4= np.fmin(dis_level_ok, vel_level_slow)
rule5= np.fmin(dis_level_ok, vel_level_ok)
rule6= np.fmin(dis_level_ok, vel_level_fast)
rule7= np.fmin(dis_level_far, vel_level_slow)
rule8= np.fmin(dis_level_far, vel_level_ok)
rule9= np.fmin(dis_level_far, vel_level_fast)
print(rule1, rule2,rule3, rule4, rule5, rule6, rule7, rule8, rule9)

decelerate_tot= max(rule1, rule2, rule3, rule6, rule9)
const_tot= max(rule4, rule5, rule8)
accelerate_tot = rule7
#print("results are:", decelerate_tot, const_tot, accelerate_tot)


fast_activation= np.fmin(vel_level_fast, vel_fast)
ok_activation= np.fmin(vel_level_ok, vel_ok)
slow_activation= np.fmin(vel_level_slow, vel_slow)

close_activation= np.fmin(dis_level_close, dis_close)
accept_activation= np.fmin(dis_level_ok, dis_ok)
far_activation= np.fmin(dis_level_far, dis_far)

decelerate_activation= np.fmin(decelerate_tot, decelerate)
const_activation= np.fmin(const_tot, const)
accelerate_activation= np.fmin(accelerate_tot, accelerate)
#print("results are:", decelerate_activation)

tip0 = np.zeros_like(x_vel)
tip1 = np.zeros_like(x_dis)
tip2 = np.zeros_like(x_acc)



#####################################
# Visualize this:
fig, (ax1, ax0) = plt.subplots(nrows=2, figsize=(8, 6))

ax0.fill_between(x_vel, tip0, fast_activation, facecolor='r', alpha=0.7)
ax0.plot(x_vel, vel_fast, 'r', linewidth=0.5, linestyle='--', label='fast')
ax0.fill_between(x_vel, tip0, ok_activation, facecolor='g', alpha=0.7)
ax0.plot(x_vel, vel_ok, 'g', linewidth=0.5, linestyle='--', label='ok')
ax0.fill_between(x_vel, tip0, slow_activation, facecolor='b', alpha=0.7)
ax0.plot(x_vel, vel_slow, 'b', linewidth=0.5, linestyle='--', label='slow')
#ax0.set_title('Car speed')
ax0.set_xlabel('Delta speed [cm/s]')
ax0.set_ylabel('Degree of membership')
ax0.legend()

ax1.fill_between(x_dis, tip1, far_activation, facecolor='b', alpha=0.7)
ax1.plot(x_dis, dis_far, 'b', linewidth=0.5, linestyle='--', label='Far')
ax1.fill_between(x_dis, tip1, accept_activation, facecolor='g', alpha=0.7)
ax1.plot(x_dis, dis_ok, 'g', linewidth=0.5, linestyle='--', label='ok')
ax1.fill_between(x_dis, tip1, close_activation, facecolor='r', alpha=0.7)
ax1.plot(x_dis, dis_close, 'r', linewidth=0.5, linestyle='--', label='Close')
#ax1.set_title('Car distance')
ax1.set_xlabel('Delta Distance [cm]')
ax1.set_ylabel('Degree of membership')
ax1.legend()

# Turn off top/right axes
for ax in (ax0,ax1,ax2,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.show()



# Turn off top/right axes
for ax in (ax0,ax1,ax2,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
plt.show()

####################################################################

# Aggregate all three output membership functions together
aggregated = np.fmax(decelerate_activation, np.fmax(const_activation, accelerate_activation))
#print(aggregated)

# Calculate defuzzified result
acc = fuzz.defuzz(x_acc, aggregated, 'centroid')
acc_activation = fuzz.interp_membership(x_acc, aggregated, acc)  # for plot
print(acc)


# Visualize this

fig, ax2 = plt.subplots(figsize=(8, 5))
ax2.fill_between(x_acc, tip2, decelerate_activation, facecolor='b', alpha=0.7)
ax2.plot(x_acc, decelerate, 'b', linewidth=0.5, linestyle='--',label='Decelerate' )
ax2.fill_between(x_acc, tip2, const_activation, facecolor='g', alpha=0.7)
ax2.plot(x_acc, const, 'g', linewidth=0.5, linestyle='--', label='Const')
ax2.fill_between(x_acc, tip2, accelerate_activation, facecolor='r', alpha=0.7)
ax2.plot([acc, acc], [0, acc_activation], 'k', linewidth=1.5, alpha=0.9)

ax2.plot(x_acc, accelerate, 'r', linewidth=0.5, linestyle='--', label='Accelerate')
#ax2.set_title('Output membership activity')
ax2.set_xlabel('Delta Acceleration [cm/s' + u"\u00B2" + ']')
ax2.set_ylabel('Degree of membership')
ax2.legend()



fig, ax0 = plt.subplots(figsize=(8, 5))

ax0.plot(x_acc, decelerate, 'r', linewidth=0.5, linestyle='--', )
ax0.plot(x_acc, const, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_acc, accelerate, 'b', linewidth=0.5, linestyle='--')
ax0.fill_between(x_acc, tip2, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([acc, acc], [0, acc_activation], 'k', linewidth=1.5, alpha=0.9)
#ax0.set_title('Aggregated membership and result (line)')
ax0.set_xlabel('Delta Acceleration [cm/s' + u"\u00B2" + ']')
ax0.set_ylabel('Degree of membership')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.show()