%function[n_colonies] = IDcfu_datapixel_waterFinal(day0, plateN, fileimage)
    %This script identifies bacteria colonies in a petri dish. It generates a
    %mask first to avoid the outer part of the plate. Later, segmentation and
    %label identification allows to target even those colonies that are not
    %completely circular. It retrieves a .jpg file with the identifies colonies
    %and a .mat file with a label, morphological and RGB pixel information.

    %clear
%%
    %to test without function
%     day0 = '190921';
%     plateN = '1';
%     fileimage = 'd30-10µl-nr5';
%     
    %Pixel transformation from DistancePix to cm
    %pixel_size=1/DistancePix;
    %1inch x 96 pixels; 1inch = 2.54cm
    pixel_size=2.54/96; %cm

    %Information required to generate a label
    day = day0; 
    %Replace plateN it with n counter
    plate = plateN;

    %Read files. Batch of pictures from a plate taken along several days
    file= fileimage;
    %16-bit file, RGB range 0-2500
    I = imread(file, 'tif');
    %Turn into 8-bit, RGB range 0-255 with a simple division
    I = uint8(I/257);

    %%
    %%Remove uninterested region from the image 
    %Get the image size to remove the rim of the petri dish
    imageSize = size(I);
    %center and radius of circle ([c_col, c_row, r]). I set my center at
    %[y0,x0] = [1040, 1015] and r = 845
    ci = [1040, 1015, 845];   
    %Make a grid the same dimensions as the original image
    [xx,yy] = ndgrid((1:imageSize(1))-ci(1),(1:imageSize(2))-ci(2));
    %Make a mask that will turn black all the area outside the plate by
    %fitting a circle with the size of the plate
    mask = uint8((xx.^2 + yy.^2)<ci(3)^2);
    %Generate the new image, cropped imaged after the mask is applied
    croppedImage = uint8(true(size(I)));
    croppedImage(:,:,1) = I(:,:,1).*mask;
    croppedImage(:,:,2) = I(:,:,2).*mask;
    croppedImage(:,:,3) = I(:,:,3).*mask;

%Remove comments if you want to print the crooped image
%     figure
%     imshow(croppedImage)
    
    %The red channel was the most informative one, therefore for colony identification
    %I decided to only take information from the red channel
    
    %Correct non-uniform ilumination. Adjust specific range if needed
    rgbI = imadjust(croppedImage(:,:,1), [0 0.60],[]);

%Remove comments if you want to print and save the image after  the crooped image
%     figure
%     imshow(rgbI)
%     print(strcat(filename{1},'-adjust'),'-dpng');
%     close;
    
%%
    %There are two types of colonies on the plates, ones with higher RGB values
    %than the background and some other with lower RGB values than the
    %background. I implemented a two-step process to identify all of them.
    
    %Filter bright colonies
    rgbBW = rgbI >=200;%imshow(rgbBW)
    %remove connected objects
    rgbI_nobord = imclearborder(rgbBW,8);%imshow(rgbI_nobord)
    %to fill up holes
    rgbI_noholes = imfill(rgbI_nobord,'holes');%imshow(rgbI_final)       
    %smooth object. Avoids extracting background information
    seD = strel('diamond',1);
    rgbI_final_W = imerode(rgbI_noholes,seD);
    
    
    %Find colonies using boundary.
    %B, returns an array of pixel locations
    %L, label matrix of objects
    %n, number of objects (labels)
    %[B1,L1,n1] = bwboundaries(rgbI_final,'noholes');   

    %Filter dark colonies
    rgbBW = rgbI < 50;%imshow(rgbBW)
    %remove connected objects
    rgbI_nobord = imclearborder(rgbBW,8);%imshow(rgbI_nobord)
    %rgbI_final = rgbI_nobord;
    %smooth object. Avoids extracting background information.
    seD = strel('diamond',1);
    rgbI_final_B = imerode(rgbI_nobord,seD);

    %Combine both segmented figures
    bw = (rgbI_final_W | rgbI_final_B);
%     figure
%     imshow(bw)

%Remove if you want to print and save the BW image got from the boundaries
%function
%     figure
%     imshow(BW)
%     print(strcat(filename{1},'-BW'),'-dpng');
%     close;

%%
%watershed
    %Remove small white objects from the image
    BW = ~bwareaopen(~bw, 10);
    %BW = L;
    %Get distance between white objects
    D = -bwdist(~BW);
    %Computes local maxima. Center of the colonies
    mask = imextendedmin(D,1,4);
    
%     figure
%     imshowpair(BW,mask,'blend')
%     print(strcat(file,'-blend'),'-dpng');
%     close;
    
    D2 = imimposemin(D,mask);
    LD = watershed(D2);
    BW2 = BW;
    BW2(LD == 0) = 0;
%     figure
%     imshow(BW2)
%     print(strcat(file,'-BW'),'-dpng');
%     close;
    %
    
    %%
    %New boundaries
    %[L] = bwboundaries(BW2,'noholes');
    [B,L] = bwboundaries(BW2,'noholes');

    %Remove comments if one wants to check the boundaries found
%         figure
%         imshow(L)
%         hold on
%         for k = 1:length(B)
%            boundary = B{k};
%            plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
%         end

%%
    %get morphological stats
    stats=  regionprops(L, 'Centroid', 'Area', 'EquivDiameter', 'Perimeter', 'Circularity', 'Eccentricity');
    Centroid = floor(cat(1,stats.Centroid));
    Eccentricity = cat(1,stats.Eccentricity);
    Area = cat(1,stats.Area);
    Diameter = cat(1,stats.EquivDiameter);
    
    %%
    %Get pixel information per channel. Use the BW image generated using
    %boundaries
    for k = 1:3
        rgbI = I(:,:,k);
        %Save according to RGB channel
        if k == 1
            %Better to have a different name for the different structures
            statsPixel_red = regionprops(L,rgbI,{'Centroid','PixelValues','MeanIntensity'});
            %save(strcat(filename{1},'-pixel_red.mat'), 'statsPixel_red');
        elseif k == 2
            %Better to have a different name for the different structures
            statsPixel_green = regionprops(L,rgbI,{'Centroid','PixelValues', 'MeanIntensity'});
            %save(strcat(filename{1},'-pixel_green.mat'), 'statsPixel_green');
        else
            %Better to have a different name for the different structures
            statsPixel_blue = regionprops(L,rgbI,{'Centroid','PixelValues', 'MeanIntensity'});
            %save(strcat(filename{1},'-pixel_blue.mat'), 'statsPixel_blue');
        end

    end

    %%
    %Filter 'bad quality data'
    %Filter by area bigger than n and eccentricity, the closes to 1 the more
    %line segment, the closes to 0 the more circular shape
    filter1 = find(Area > 350 & Area < 70000 & Eccentricity < 0.75); 

    %Find the elements close the the petri dish walls
    filter2 = zeros(length(filter1),1);

    %Calculate indexes to plot a circle and compare with the indexes in mask
    th = 0:pi/5:2*pi;
    for i = 1:length(filter1)
        %imshow(croppedImage);
        %Fit a circle
        xunit = floor(Diameter(filter1(i))/2 * cos(th) + Centroid(filter1(i),1));
        yunit = floor(Diameter(filter1(i))/2 * sin(th) + Centroid(filter1(i),2));
        %Find within the boundaries. Check ci variable
        %mean_xy = mean(yunit>200 & yunit<1850 & xunit>200 & xunit<1850);
        mean_xy = mean(yunit<250 | yunit>1825 | xunit<250 | xunit>1800);
        if mean_xy > 0 
            if Area(filter1(i)) < 1000 || Eccentricity(filter1(i)) > 0.35
            %&& Area(filter1(i)) < 300 && 
            filter2(i) = filter1(i);
            end 
        end
    end

    filter2 = nonzeros(filter2);
    filter2 = setdiff(filter1, filter2);
    
    %%
    %Get std on pixel values of the selected colonies
    std_red = zeros(1, length(filter2));
    std_green = zeros(1, length(filter2));
    std_blue = zeros(1, length(filter2));
    
    %Go to each of the colonies that past the two filters and get the std
    %values using the information from PixelValues
    for m = 1:length(filter2)
        std_red(m) = std(double(statsPixel_red(filter2(m)).PixelValues));
        std_green(m) = std(double(statsPixel_green(filter2(m)).PixelValues));
        std_blue(m) = std(double(statsPixel_blue(filter2(m)).PixelValues));    
    end

    %%
    %Get data filtered 
    diameter = floor(Diameter(filter2));
    centroid = Centroid(filter2,:);
    area = cat(1,stats(filter2).Area);
    perimeter = cat(1,stats(filter2).Perimeter);
    circularity = cat(1,stats(filter2).Circularity);
    eccentricity = cat(1,stats(filter2).Eccentricity);
    R_mean = cat(1,statsPixel_red(filter2).MeanIntensity);
    G_mean = cat(1,statsPixel_green(filter2).MeanIntensity);
    B_mean = cat(1,statsPixel_blue(filter2).MeanIntensity);


    %%Plot the colonies found and save data
    %At this step labels are generated
    colony = strings(length(diameter),1); 
    label = strings(length(diameter),1);
    ID = strings(length(diameter),1);
    %Give labels
    for i = 1:length(diameter)
        colony(i) = int2str(i);
        label(i)= strcat(day,'-',plate,'-',int2str(i));
        ID(i)=strcat(plate,'-',int2str(i));
    end
    
    % 
    figure
    imshow(I);
    viscircles(centroid,diameter/2,'EdgeColor','b');
    text(centroid(:,1), centroid(:,2), colony);
    print(strcat(file,'-IDs_Water'),'-dpng');
    close;
    %%
%     Check how to save the data, all the variables bwboundaries, regiongprops
%     morphological and pixel values
%     Save data
    statsData = struct('label', label, 'ID', ID, 'centroid', centroid,...
        'area', area, 'diameter', diameter,'perimeter', perimeter, 'circularity', circularity,...
        'eccentricity', eccentricity, 'R_mean', R_mean, 'R_std', std_red,...
        'G_mean', G_mean, 'G_std', std_green, 'B_mean', B_mean, 'B_std', std_red);
    save(strcat(file,'-dataWater.mat'), 'statsData');
    
    n_colonies = length(area);
%end